"""Repack this pinned external-FP16 Flow as float or MNN IDST W8 candidates.

No converter dependency: follows MNN 3.6.1 IDSTEncoder dense asymmetric W8
format and leaves all graph operations unchanged. This is NOT HQQ and does
not fit activation scales. Corpus replay is required before any deployment.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import struct
import re

from audit_mnn_weight_layout import FlatBuffer, audit
from flow_quant_common import digest

FLOW = "flow.cfg-student-2step.batch1.fp16.mnn"
GROUPS={
    'ffn-w8':set(), 'ffn-q-w8':{'q'}, 'ffn-k-w8':{'k'},
    'ffn-v-w8':{'v'}, 'ffn-out-w8':{'out'},
    'ffn-qout-w8':{'q','out'}, 'ffn-qvout-w8':{'q','v','out'},
    'ffn-qkv-w8':{'q','k','v'}, 'ffn-qkvo-w8':{'q','k','v','out'},
}

def projection_group(name: str) -> str | None:
    match=re.match(r'^/transformer_blocks\.(\d+)/attn/(to_q|to_k|to_v|to_out\.0)/',name)
    if not match or not 0<=int(match[1])<22:return None
    return {'to_q':'q','to_k':'k','to_v':'v','to_out.0':'out'}[match[2]]

def selected_operator(name: str,mode: str) -> bool:
    if mode=='fp32':return False
    if mode=='all-w8':return True
    if mode not in GROUPS:raise ValueError('Unknown reviewed projection recipe')
    return '/ff/' in name or projection_group(name) in GROUPS[mode]


def clear_table_field(data: bytearray, table: int, slot: int) -> None:
    """Private vtable copy: shared vtables elsewhere must never be modified."""
    original = table - struct.unpack_from("<i", data, table)[0]
    size = struct.unpack_from("<H", data, original)[0]
    if 4 + slot * 2 >= size: raise ValueError("Expected field not in vtable")
    replacement = bytearray(data[original:original+size])
    struct.pack_into("<H",replacement,4+slot*2,0)
    while len(data)%2: data.append(0)
    position=len(data); data.extend(replacement)
    # FlatBuffers signed vtable displacement supports deduplicated vtables
    # after the table, too. Ordinary object/vector offsets remain forward.
    struct.pack_into("<i",data,table,table-position)


def append_quant_table(data: bytearray, rows: int, bits: int = 8) -> int:
    # IDSTQuan slots 0..11. External buffer and alpha vectors are absent.
    while len(data)%4: data.append(0)
    vtable=len(data)
    offsets=[0,0,4,0,8,0,0,12,16,20,0,24]
    data.extend(struct.pack("<HH"+"H"*12,28,28,*offsets))
    table=len(data);data.extend(bytes(28))
    struct.pack_into("<i",data,table,table-vtable)
    struct.pack_into("<i",data,table+4,1)  # IDST dense quantization
    struct.pack_into("<f",data,table+8,1.0)
    if bits not in (4,8):raise ValueError('Only reviewed W4/W8 codecs')
    struct.pack_into("<iii",data,table+12,bits,-(1<<(bits-1)),rows)
    data[table+24]=1  # uint32 shape header
    return table


def encode_w8(weights, output_channels: int, block: int):
    import numpy as np
    shaped=np.asarray(weights,dtype=np.float32).reshape(output_channels,-1)
    width=shaped.shape[1]
    if width%block: raise ValueError("Block does not divide selected linear input")
    grouped=shaped.reshape(output_channels,width//block,block)
    minimum=grouped.min(axis=-1)
    scale=(grouped.max(axis=-1)-minimum)/np.float32(255)
    good=scale>np.float32(1e-6)  # Same gate as MNN IDSTEncoder::WriteCQBlobs
    denominator=np.where(good,scale,np.float32(1))
    indices=np.floor((grouped-minimum[...,None])/denominator[...,None]+np.float32(.5))
    indices=np.where(good[...,None],indices,0).clip(0,255).astype(np.uint8)
    alpha=np.stack((minimum,scale),axis=-1).astype("<f4")
    rows=minimum.size
    header=bytes([2])+struct.pack("<II",rows,block)+bytes([0])+bytes(range(128,256))+bytes(range(128))
    # Decoder converts stored indices to signed q, then adjusts asymmetric
    # minima by +128*scale. Keep that operation order for weight-error checks.
    restored=((indices.astype(np.float32)-np.float32(128))*scale[...,None]+
              (minimum+np.float32(128)*scale)[...,None])
    error=float(np.linalg.norm((restored-grouped).astype(np.float64))/max(1e-30,np.linalg.norm(grouped.astype(np.float64))))
    return header+indices.tobytes(),alpha.tobytes(),rows,error


def external_regions(f: FlatBuffer, weight_bytes: int) -> list[dict]:
    net=f.number("I",0)
    if f.table_vector(net,10): raise ValueError("Subgraphs need a separately reviewed repacker")
    regions=[]
    for op in f.table_vector(net,3):
        kind=f.scalar(op,1,fmt="B")
        slot={9:6,7:9,88:5}.get(kind) # Conv2D, Blob, LayerNorm
        if slot is None: continue
        main=f.pointer(op,2);external=f.int64_vector(main,slot)
        if not external:continue
        p=f.pointer(op,3)
        name=f.data[p+4:p+4+f.number("I",p)].decode()
        regions.append({"kind":kind,"main":main,"vector":f.pointer(main,slot),"external":external,"name":name})
    regions.sort(key=lambda x:x["external"][0])
    position=0
    for region in regions:
        values=region["external"]
        if values[0]!=position or any(x<0 for x in values):raise ValueError("External regions overlap or leave unknown gaps")
        position+=sum(values[1:])
    if position!=weight_bytes:raise ValueError("Not all source external bytes accounted for")
    if len([r for r in regions if r["kind"]==9])!=191:raise ValueError("Unexpected Flow convolution layout")
    return regions


def repack(source: Path, destination: Path, mode: str, block: int, bits: int = 8, algorithm: str = 'rtn', w4_scope: str = 'all-selected') -> dict:
    import numpy as np
    if mode not in (*GROUPS,'fp32','all-w8') or block not in (32,64,128):raise ValueError('Unknown recipe/block')
    if bits not in (4,8) or algorithm not in ('rtn','hqq') or w4_scope not in ('all-selected','ffn','projections'):raise ValueError('Unreviewed PTQ configuration')
    if bits==4 and mode not in ('ffn-w8','ffn-qvout-w8'):raise ValueError('W4 screening keeps K and other operators original')
    if destination.exists():raise FileExistsError(destination)
    source_graph=source/FLOW;source_weight=source/(FLOW+".weight")
    original=source_graph.read_bytes();f=FlatBuffer(original);graph=bytearray(original)
    regions=external_regions(f,source_weight.stat().st_size)
    if mode in GROUPS:
        names=[r['name'] for r in regions if r['kind']==9]
        if sum('/ff/' in name for name in names)!=44:raise ValueError('Expected 44 FFN linear operators')
        for group in GROUPS[mode]:
            chosen=[name for name in names if projection_group(name)==group]
            if len(chosen)!=22 or len(set(chosen))!=22:raise ValueError('Expected 22 distinct '+group+' projections')
    hashes={"graph":digest(source_graph),"weight":digest(source_weight)}
    # Exact graph used by the frozen local 20+5 corpus. It differs from the
    # older published phone manifest; do not silently relabel either identity.
    if hashes!={"graph":"b74ece0a1a2e165eef53dbc8d628310a1b464323d4729959069744baecf418d7",
                "weight":"e8d363544063702a9da0800f3a2bbba5669ea719943a5de15a8b08b0f28d24bc"}:
        raise ValueError("Repacker only accepts this reviewed FP16 Flow export")
    destination.mkdir(parents=True,exist_ok=False)
    output_graph=destination/"flow.mnn";output_weight=destination/"flow.mnn.weight"
    rows=[]
    with source_weight.open("rb") as reader,output_weight.open("xb") as writer:
        for region in regions:
            e=region["external"];main=region["main"]
            reader.seek(e[0]);old=reader.read(sum(e[1:]));offset=writer.tell()
            if region["kind"]!=9:
                writer.write(old);updated=[offset,*e[1:]]
            else:
                q=f.pointer(main,3);common=f.pointer(main,0)
                if not q or f.scalar(q,2)!=3 or len(e)!=5 or e[2]!=0 or e[4]!=0:
                    raise ValueError("Expected plain external FP16 convolution")
                oc,ic=f.scalar(common,10),f.scalar(common,11)
                area=f.scalar(common,2,1)*f.scalar(common,3,1)
                groups=f.scalar(common,9,1)
                weight=np.frombuffer(old[:e[1]],dtype="<f2").astype("<f4")
                if weight.size!=oc*ic*area//groups or not np.isfinite(weight).all():raise ValueError("Weight dimensions/finite check failed")
                bias=old[e[1]:]
                selected=selected_operator(region['name'],mode)
                if selected:
                    if area!=1 or groups!=1:
                        if mode=="all-w8": selected=False
                        else: raise ValueError("Selected FFN is not a plain linear")
                if selected:
                    use_w4=bits==4 and (w4_scope=='all-selected' or (w4_scope=='ffn' and '/ff/' in region['name']) or (w4_scope=='projections' and projection_group(region['name']) in {'q','v','out'}))
                    if use_w4:
                        from flow_w4_ptq import encode_w4
                        payload,alpha,quant_rows,error=encode_w4(weight,oc,block,algorithm)
                    else:payload,alpha,quant_rows,error=encode_w8(weight,oc,block)
                    writer.write(payload);writer.write(alpha);writer.write(bias)
                    updated=[offset,len(payload),len(alpha),len(bias),0]
                    table=append_quant_table(graph,quant_rows,4 if use_w4 else 8)
                    field=f.field(main,3);struct.pack_into("<I",graph,field,table-field)
                    kind="w4" if use_w4 else "w8"
                elif mode=="fp32":
                    payload=weight.tobytes();writer.write(payload);writer.write(bias)
                    updated=[offset,len(payload),len(bias)]
                    clear_table_field(graph,main,3)
                    kind="fp32";error=0
                else:
                    writer.write(old);updated=[offset,*e[1:]];kind="original-fp16";error=0
                rows.append({"name":region["name"],"values":int(weight.size),"format":kind,"weight_relative_l2":error,
                    "group":'ffn' if '/ff/' in region['name'] else projection_group(region['name']) or 'preserved',
                    "output_channels":oc,"input_channels":ic,"kernel_area":area,"groups":groups})
            vector=region["vector"]
            struct.pack_into("<I",graph,vector,len(updated))
            for index,value in enumerate(updated):struct.pack_into("<q",graph,vector+4+index*8,value)
    output_graph.write_bytes(graph)
    report={"mode":mode,"block":block,"source_hashes":hashes,"source_weight_bytes":source_weight.stat().st_size,
            "candidate_weight_bytes":output_weight.stat().st_size,"graph_sha256":digest(output_graph),
            "weight_sha256":digest(output_weight),"operators":rows,"layout_audit":audit(output_graph),
            "quantized_operator_count":sum(row['format'] in ('w4','w8') for row in rows),
            "note":("Decoded FP16 to exact FP32 values, removed encoded-quant metadata; replay acceptance still required" if mode=="fp32" else
                    "Native MNN-compatible dense asymmetric min-max W8; not HQQ, no activation calibration fit, replay acceptance still required")}
    if bits==4:
        report.update({'bits':bits,'algorithm':algorithm,'w4_scope':w4_scope,'w4_operators':sum(r['format']=='w4' for r in rows),
          'w8_operators':sum(r['format']=='w8' for r in rows),'note':'Native IDST W4 packed nibbles, '+algorithm+' weight-only; K and nonselected bytes unchanged; no activation-fit/GPTQ/AWQ; not phone acceptance'})
    (destination/"repack-report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("models/CosyVoice3"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--mode",choices=(*GROUPS,"fp32","all-w8"),required=True)
    parser.add_argument("--block",type=int,choices=(32,64,128),default=128)
    args=parser.parse_args();report=repack(args.source.resolve(),args.output.resolve(),args.mode,args.block)
    print(json.dumps({k:v for k,v in report.items() if k!="operators"},indent=2))
