PY ?= python
RUN = PYTHONPATH=src $(PY) -m complaint_triage.cli
N ?= 500

prepare:  ; $(RUN) prepare --input data/raw/complaints.csv
baseline: ; $(RUN) baseline
llm:      ; $(RUN) llm --mode zeroshot --n $(N) && $(RUN) llm --mode fewshot --n $(N)
compare:  ; $(RUN) compare
gold:     ; $(RUN) gold-sample --n 150
gold-eval:; $(RUN) gold-eval --gold gold/annotation_labeled.csv
test:     ; PYTHONPATH=src $(PY) -m pytest -q
all: prepare baseline llm compare
.PHONY: prepare baseline llm compare gold gold-eval test all
