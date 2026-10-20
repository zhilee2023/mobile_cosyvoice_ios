"""Replay real two-step fixtures: teacher-forced velocity and closed-loop Mel.

Windows CPU correctness/sensitivity, not iPhone RTF. All six inputs are copied
before every execution. Held-out texts never enter a calibration fit.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import wave

from flow_quant_common import digest


def metric(actual,expected):
    import numpy as np
    a=actual.astype(np.float64);b=expected.astype(np.float64)
    return {"relative_l2":float(np.linalg.norm(a-b)/max(1e-30,np.linalg.norm(b))),
            "max_absolute_error":float(np.max(np.abs(a-b))),"finite":bool(np.isfinite(actual).all())}


def evaluate(model:Path,corpus:Path,destination:Path,split:str,limit:int,memory:str,audio:bool,assets:Path|None=None):
    import MNN
    import numpy as np
    from pc_flow_hift_probe import model_session,put,get,hift
    from mnn_session_config import cpu_session_config
    destination.mkdir(parents=True,exist_ok=False)
    reference=json.loads((corpus/"calibration-report.json").read_text(encoding="utf-8"))
    if reference["state"]!="completed" or not reference.get("verification",{}).get("real_two_step_trajectories_exact"):
        raise ValueError("Use a completed and verified corpus")
    cases=[c for c in reference["cases"] if split=="all" or c["split"]==split]
    if limit:cases=cases[:limit]
    recipe_path=model.parent/"repack-report.json"
    recipe=json.loads(recipe_path.read_text()) if recipe_path.is_file() else None
    if recipe:
        source_files=reference["model_files"]
        if recipe["source_hashes"]!={"graph":source_files["flow.cfg-student-2step.batch1.fp16.mnn"]["sha256"],
                                     "weight":source_files["flow.cfg-student-2step.batch1.fp16.mnn.weight"]["sha256"]}:
            raise ValueError("Candidate does not originate from corpus reference model")
    report={"state":"running","model":str(model),"model_sha256":digest(model),
            "weight_sha256":digest(Path(str(model)+".weight")),"recipe":recipe,
            "scope":"Windows CPU High, 4 threads; not iPhone or whole-chain RTF",
            "memory_mode":memory,"split":split,"cases":[]}
    report["session_config"]=cpu_session_config("high",memory,4)
    report["binding_memory_enum_verified"]=True
    def save(): (destination/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    save()
    try:
        loaded=time.perf_counter()
        network,session=model_session(model,precision="high",memory=memory)
        report["session_creation_seconds"]=time.perf_counter()-loaded
        last_shapes=None
        for case in cases:
            print(f"{model.parent.name}/{memory}: {case['id']} starts",flush=True)
            arrays=[]
            for snapshot in case["snapshots"]:
                with np.load(Path(snapshot["directory"])/"tensors.npz") as archive:
                    arrays.append({name:archive[name].copy() for name in archive.files})
            shapes={name:value.shape for name,value in arrays[0].items() if name!="estimator_out"}
            setup=time.perf_counter()
            if shapes!=last_shapes:
                for name,shape in shapes.items():network.resizeTensor(network.getSessionInput(session,name),tuple(shape))
                network.resizeSession(session);last_shapes=shapes
            setup_seconds=time.perf_counter()-setup
            def forward(inputs):
                for name in shapes:put(network,session,name,inputs[name])
                started=time.perf_counter()
                code=network.runSession(session)
                if code!=0:raise RuntimeError(f"MNN runSession error {code}")
                value=get(network,session,"estimator_out").reshape(arrays[0]["x"].shape)
                if not np.isfinite(value).all():raise RuntimeError("Nonfinite Flow output")
                return value,time.perf_counter()-started
            v1,t1=forward(arrays[0]);v2,t2=forward(arrays[1])
            closed=dict(arrays[1]);closed["x"]=arrays[0]["x"]+np.float32(case["snapshots"][0]["delta"])*v1
            if np.array_equal(closed["x"],arrays[1]["x"]):
                vclosed,tc=v2,t2
            else:vclosed,tc=forward(closed)
            final=closed["x"]+np.float32(case["snapshots"][1]["delta"])*vclosed
            mel=final[0,:,174:case["effective_frames"]].copy()
            original=np.load(Path(case["audio"]).parent/"target-mel.npy")
            row={"id":case["id"],"split":case["split"],"audio_seconds":case["audio_seconds"],
                 "frames":case["runtime_frames"],"resize_seconds":setup_seconds,
                 "teacher_velocity":[metric(v1,arrays[0]["estimator_out"]),metric(v2,arrays[1]["estimator_out"])],
                 "closed_loop_mel":metric(mel,original),"two_step_run_read_seconds":t1+tc,
                 "teacher_step_run_read_seconds":[t1,t2],"closed_loop_step2_run_read_seconds":tc}
            row["runtime_session_memory_mb"]=network.getSessionInfo(session,0)
            np.save(destination/f"{case['id']}-mel.npy",mel)
            if audio:
                source=assets or Path(__file__).resolve().parents[1]/"models/CosyVoice3"
                pcm,times=hift(source,mel,core_precision="high",core_memory="normal")
                if not np.isfinite(pcm).all():raise RuntimeError("Nonfinite candidate PCM")
                path=destination/f"{case['id']}.wav"
                with wave.open(str(path),"wb") as stream:
                    stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(24000)
                    stream.writeframes((np.clip(pcm,-1,1)*32767).astype("<i2").tobytes())
                with wave.open(case["audio"],"rb") as stream: baseline=np.frombuffer(stream.readframes(stream.getnframes()),dtype="<i2").astype(np.float32)/32767
                from audio_sensitivity_metrics import read_wav,compare,correlation
                baseline_wav=read_wav(Path(case["audio"]));candidate_wav=read_wav(path)
                row["pcm"]=metric(candidate_wav,baseline_wav)
                row["pcm"]["correlation"]=correlation(baseline_wav,candidate_wav)
                row["audio_spectrum"]=compare(baseline_wav,candidate_wav)
                row["wav"]=str(path);row["hift_seconds"]=times
            report["cases"].append(row);save()
            print(f"{case['id']}: velocity L2 {row['teacher_velocity'][0]['relative_l2']:.6g}/{row['teacher_velocity'][1]['relative_l2']:.6g}; closed Mel L2 {row['closed_loop_mel']['relative_l2']:.6g}",flush=True)
        report["state"]="completed"
        report["summary"]={"cases":len(report["cases"]),"max_teacher_velocity_l2":max(m["relative_l2"] for row in report["cases"] for m in row["teacher_velocity"]),
                           "max_closed_loop_mel_l2":max(row["closed_loop_mel"]["relative_l2"] for row in report["cases"]),
                           "mean_closed_loop_mel_l2":float(np.mean([row["closed_loop_mel"]["relative_l2"] for row in report["cases"]]))}
    except Exception as error:
        report["state"]="failed";report["error"]=str(error);save();raise
    save();return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True);parser.add_argument("--corpus",type=Path,default=Path("builds/flow-calibration-20261020"))
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--split",choices=("all","calibration","validation"),default="validation")
    parser.add_argument("--limit",type=int,default=0);parser.add_argument("--memory",choices=("normal","low"),default="normal");parser.add_argument("--audio",action="store_true")
    parser.add_argument('--assets',type=Path,help='Original HiFT assets for listening candidates, read-only')
    args=parser.parse_args();r=evaluate(args.model.resolve(),args.corpus.resolve(),args.output.resolve(),args.split,args.limit,args.memory,args.audio,args.assets.resolve() if args.assets else None)
    print(json.dumps(r.get("summary",{"state":r["state"]}),indent=2))
