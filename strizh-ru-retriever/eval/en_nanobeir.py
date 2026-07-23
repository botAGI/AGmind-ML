import os, time
os.environ["HF_HUB_DISABLE_XET"]="1"
import mteb
from sentence_transformers import SentenceTransformer
m=SentenceTransformer(os.path.expanduser("~/strizh/strizh-embed-4L-v2"), device="cuda"); m.max_seq_length=512
tasks=mteb.get_benchmark("NanoBEIR")
out=os.path.expanduser("~/strizh/en_strizh-v2")
for i,t in enumerate(tasks,1):
    for att in range(1,4):
        try:
            mteb.MTEB(tasks=[t]).run(m,output_folder=out,encode_kwargs={"batch_size":64},verbosity=0)
            print(f"OK [{i}/{len(tasks)}] {t.metadata.name}",flush=True); break
        except Exception as e:
            if att==3: print(f"FAIL {t.metadata.name}",flush=True)
            else: time.sleep(15*att)
print("EN_V2_DONE",flush=True)
