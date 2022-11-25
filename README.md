# Summary
Streamlit landing page that redirects from NLP ML Toolkit intro page - https://ml-interpretable.herokuapp.com/ *Note: site not updated regularly*.

# Features

* Embedding Evaluation - Compare how spread out/well-defined different word embeddings are. This helps with word sense disambiguation if the clusters of embeddings between different words are well-defined.

* Polar Opposite Visualization - Compare the effects of polar opposite directional vector transformations on different feature extraction embeddings for interpretability.

# Conclusions

* The tool reveals that whether or not polar opposite embedding dimensions are well-defined (Alaska having a higher absolute coefficient for cold-hot dimension) depends on the original embeddings. We may want to look into what makes certain embeddings better than others at having more pronounced polar opposite dimensions. For instance, the facebook word embedding in the tool is much better at having a specific polar opposite associated with a term than the roberta embeddings.
