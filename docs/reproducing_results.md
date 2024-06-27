# Reproducing the Results

For reproducing the model result, use the snippet below. The configs are provided in the `data/inputs/configs/` folder.


```python
from clinical_ner.models import SpanExtractor
from clinical_ner.evaluation import Evaluator
from clinical_ner.benchmarks import MEDICS_NER

model_name = "alvaroalon2/biobert_diseases_ner"

benchmark = MEDICS_NER 

# the below config is model and dataset specific. To reproduce the results, load the config from the config folder
dataset_wise_config = # load config
model = SpanExtractor.from_predefined(model_name)

evaluator = Evaluator(model, benchmark=benchmark, dataset_wise_config=dataset_wise_config)
evaluation.run()
```

