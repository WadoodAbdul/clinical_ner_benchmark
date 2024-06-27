
# Custom Model Implementation

We have implementation for NER models that are based on 
1. HuggingFace encoder models - in essence, token classifiers
2. HuggingFace decoder models - autoregressive LMs, these also need a prompt template
3. GLiNER based models


In case your model doesn't fit any of the model classes above, you can implement a custom model.

To implement a custom model, we have to ensure the CustomModel inherits from the `GenericSpanExtractor` or `SpanExtractor` abstract classes.

```python
from medics_ner.models import GenericSpanExtractor
from medics_ner.models.span_dataclasses import NERSpans

class MyCustomModel(GenericSpanExtractor):
    def extract_spans_from_chunk(text: str, **kwargs) -> NERSpans:
        """
        Extracts spans from sequences of any length

        Args:
            text: The text from which spans should be extracted.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The NERSpans.
        """
        pass


model = MyModel()
benchmark = MEDICS_NER 

# the below config is model and dataset specific.
dataset_wise_config = {
        "dataset_name": {"label_normalization_map": {"DISEASE": "condition"}}
    }
evaluator = Evaluator(model, benchmark=benchmark, dataset_wise_config=dataset_wise_config)
evaluation.run()
```

Feel free to open an issue or reach out to our team for any assistance!

