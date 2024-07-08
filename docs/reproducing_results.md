# Reproducing the Results

The leaderboard results can be reproduced using the code below to ensure scientific rigour.

## Single model

For reproducing the model result, use the snippet below. The configs are provided in the `data/inputs/inference_configs/` folder.

```python
from clinical_ner.models import SpanExtractor
from clinical_ner.evaluation import Evaluator
from clinical_ner.benchmarks import MEDICS_NER
from clinical_ner.leaderboard import get_leaderboard_models

model_name = "alvaroalon2/biobert_diseases_ner"

benchmark = MEDICS_NER 

# the below config is model and dataset specific. To reproduce the results, we load the config from the inference_configs folder
model, dataset_wise_config = load_leaderboard_model_and_config(model_name)

evaluator = Evaluator(model, benchmark=benchmark, dataset_wise_config=dataset_wise_config)
evaluator.run()
```

## Complete leaderboard

```python

from clinical_ner.leaderboard import get_leaderboard_models, reproduce_leaderboard_results

# loads the list of existing models on the leaderboard
leaderboard_models = get_leaderboard_models()

# runs the same evaluation script used for the benchmark and stores the results
reproduce_leaderboard_results(models=leaderboard_models,
                                output_dir='./reproduced_results')

```

the snippet above saves the result in the `reproduced_results` folder.