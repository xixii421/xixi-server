"""Persistent full-model baseline with the same preprocessing and HTTP adapter."""
import gc
import os
import threading
import time
import uuid

import torch

from .vpd_local import (load_full_model,load_processor,prepare_inputs,prepare_request,language_forward)


class MonolithicVLM:
    scope="single-process full-model baseline; same preprocessing and greedy decoding"

    def __init__(self,spec):
        self.spec=spec
        self.processor=load_processor(spec)
        self.model=None
        self.processes=[]
        self._closed=False
        self._lock=threading.Lock()

    def __enter__(self):
        self.model=load_full_model(self.spec).to(self.spec.device)
        return self

    def prepare(self,messages,new_tokens=8):
        return prepare_request(self.spec,messages,new_tokens)

    def run(self,messages,new_tokens=8,*,on_token=None):
        return self.run_prepared(self.prepare(messages,new_tokens),on_token=on_token)

    @torch.inference_mode()
    def run_prepared(self,prepared,*,on_token=None):
        normalized,images,new_tokens=prepared.messages,prepared.images,prepared.new_tokens
        with self._lock:
            if self._closed:
                raise RuntimeError("monolithic baseline is closed")
            started=time.perf_counter()
            inputs=prepare_inputs(self.spec,self.processor,normalized,images)
            tokens=inputs["input_ids"][0].tolist()
            self.spec.validate(tokens,new_tokens)
            inputs={k:v.to(self.spec.device,dtype=getattr(torch,self.spec.dtype)) if k=="pixel_values"
                    else v.to(self.spec.device) for k,v in inputs.items()}
            result=self.model(**inputs,use_cache=True,logits_to_keep=1)
            output=[int(result.logits[0,-1].argmax())]
            times=[time.perf_counter()]
            if on_token:
                on_token(output[-1])
            for _ in range(new_tokens-1):
                result=language_forward(self.spec,self.model,[output[-1]],cache=result.past_key_values)
                output.append(int(result.logits[0,-1].argmax()))
                times.append(time.perf_counter())
                if on_token:
                    on_token(output[-1])
            return {"request_id":uuid.uuid4().hex,"mode":"monolithic_full_model",
                    "token_ids":output,"received_cache_tokens":len(tokens),
                    "token_ready_times_s":times,"image_count":len(images),"worker_pids":[os.getpid()],
                    "backend_ttft_ms":(times[0]-started)*1000,
                    "backend_tbt_ms":[(b-a)*1000 for a,b in zip(times,times[1:])],
                    "e2e_ms":(time.perf_counter()-started)*1000,
                    "vision_images_encoded":len(images),"cache_hits":0,"cache_misses":len(images),
                    "cache_resident_bytes":0,"scope":self.scope}

    def close(self):
        self._closed=True
        self.model=None
        gc.collect()
        if self.spec.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def __exit__(self,*_):
        self.close()
