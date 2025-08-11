def fuse_scores(fatigue, attention, stress, weights, min_conf=0.5):
    # Clamp by confidence: if module conf < min, downweight
    wf = weights["fatigue"] * (fatigue["conf"] if fatigue["conf"]>=min_conf else 0.25)
    wa = weights["attention"] * (attention["conf"] if attention["conf"]>=min_conf else 0.25)
    ws = weights["stress"] * (stress["conf"] if stress["conf"]>=min_conf else 0.25)
    total = max(1e-6, wf + wa + ws)
    readiness = (100 - fatigue["score"]) * (wf/total) + attention["score"] * (wa/total) + (100 - stress["score"]) * (ws/total)

    flags = []
    if fatigue["score"] > 70: flags.append("Drowsy")
    if attention["score"] < 40: flags.append("Off-task")
    if stress["score"] > 70: flags.append("High stress")

    return {"readiness": float(max(0,min(100, readiness))), "flags": flags}
