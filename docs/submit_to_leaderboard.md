# Adding a Model to the Leaderboard

To submit a model to the leaderboard:
1. Load your model using the SpanExtractor class(or implement a SpanExtractor class for your model.)
2. Run evaluation using the Evaluator class. This stores model outputs and metrics.
3. Create a PR in the m42-health/ner_leaderboard_requests (or results) dataset 
4. Your model results will be uploaded to the leaderboard!

### Getting the evaluation outputs

```python
from medics.models import SpanExtractor
from evaluation_pipeline.evaluator import Evaluator
from medics.benchmarks import MEDICS_NER

model_name = "alvaroalon2/biobert_diseases_ner"

# this is model and dataset specific.
dataset_wise_config = {
        "NCBI": {"label_normalization_map": {"DISEASE": "condition"}}
    }
# load a predefined model (or for a custom implementation see https://github.com/WadoodAbdul/medics_ner/blob/main/docs/custom_model_implementation.md)
model = SpanExtractor.from_predefined(model_name)


benchmark = MEDICS_NER # or use a specific benchmark
# or 
# tasks = mteb.get_tasks(...) # get specific tasks

evaluator = Evaluator(model, benchmark=benchmark, dataset_wise_config=dataset_wise_config)
evaluation.run()
```
This will save the results in a folder called results/{model_name}/{model_revision}. ????

### 



