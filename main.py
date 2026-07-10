from transformers import pipeline

pipe = pipeline("translation", model="ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)
