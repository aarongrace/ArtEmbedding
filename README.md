# Encoding Paintings through Real-Time Expert-Guided Embedding

## Introduction

Encoding paintings is a central challenge in both art history and artificial intelligence. Art-historical objects rarely conform to clean categorical boundaries. Movements, genres, and stylistic sensibilities are fluid, overlapping, and historically contingent.

Consider Jacques-Louis David's *Napoleon Crossing the Saint-Bernard* (1801). David is widely acknowledged as the leading French Neoclassical painter, yet this work is dramatically expressive, emotionally charged, and heroic in a manner far more aligned with Romanticism. Similar ambiguity occurs in genre classification: although primarily a historical portrait, it adopts the rhetoric of monumental narrative painting. A simple categorical label—"Neoclassical"—obscures this complexity.

Existing computational approaches typically treat art classification as a discrete labeling problem. Yet artworks often sit *between* categories, or draw simultaneously from multiple visual traditions. To represent paintings accurately, a system must acknowledge the *graded*, *overlapping*, and *continuous* nature of stylistic identity.

This project seeks to operationalize this art-historical understanding in a machine learning framework. The core aim is to construct continuous stylistic embeddings, refined through expert interpretation, that can support meaningful search, analysis, and interpretation.

## How to Run

### Prerequisites
- Python 3.8+
- Node.js and npm
- Google Chrome (for WikiArt scraper)

### 1. Preliminary Training

To train the model on the WikiArt dataset:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r req.txt

# Turn the venv into a kernel env
python -m ipykernel install --user --name=painting-embed-env --display-name "Art Embedding Env"

```
Now you can run ```regression_model.ipynb``` in the root folder

Note that the sample paintings in ```paintings/``` are only a tiny portion of the overall dataset

### 2. Web Application for Expert Annotation

To run the full-stack annotation interface:

```bash
# Install the concurrently module
cd webApp
npm install

# Install frontend dependencies
cd frontend
npm install

# Return to webApp directory and start both frontend and backend
cd ..
npm start
```

This will start the FastAPI backend and React frontend simultaneously. The annotation interface will be available in your browser.

### 3. WikiArt Scraper

To scrape paintings from WikiArt:

#### Load the Chrome extension
1. Open Chrome and go to chrome://extensions/
2. Enable "Developer mode" in the top right
3. Click "Load unpacked" and select the extension folder

#### Use the scraper
1. Navigate to a WikiArt movement page (e.g., https://www.wikiart.org/en/paintings-by-style/romanticism)
2. Click "Add All Artists" in the extension popup
3. Click "Start Download Loop" to begin scraping


## Project Status

###  Completed
- [✓] Create dataset by scraping 30,928 paintings and metadata from WikiArt.org
- [✓] Design and implement model architecture (BLIP-2 with three regression heads)
- [✓] Conduct preliminary training based on hard label database
- [✓] Create full-stack web application
- [✓] FastAPI backend with model integration for forward and backward passes
- [✓] Cache system for predictions and backwards queue
- [✓] React frontend annotation loop

### In Progress
- [ ] User registration/login system
- [ ] Multi-modal search engine for paintings
- [ ] Contrastive learning component implementation
- [ ] Focus-guided adversarial image generation and testing
- [ ] Cluster analysis

## Problem Formulation

Each painting is represented by an image *I* and mapped to an 18-dimensional continuous embedding vector **v** = [**m**, **g**, **f**] where:

- **m** ∈ ℝ⁶: Degree of association with six art movements (Baroque, Rococo, Neoclassicism, Romanticism, Realism, Impressionism)
- **g** ∈ ℝ⁶: Degree of association with six genres (Historical, Religious, Mythological, Everyday Life, Landscape, Portrait)
- **f** ∈ ℝ⁶: Formal/perceptual attributes (Balance, Complexity, Emotionality, Dynamism, Naturalism, Texture)

Each dimension is continuous, typically normalized to [0,1]. Unlike traditional classification, paintings may express high scores across multiple categories simultaneously, reflecting the true nature of art-historical complexity.

## Model Architecture

To more accurately capture the nuances of art-historical reality, this project represents paintings using a continuous, multi-dimensional embedding. Rather than assigning discrete labels, each painting receives a vector of scores for movements, genres, and formal attributes. For example, a work might be 0.7 Romantic, 0.4 Neoclassical, 0.3 History Painting, and 0.5 Everyday Scene. This allows the model to reflect that individual works can straddle movements, incorporate multiple genres, or exhibit intermediate stylistic qualities, producing a richer and more flexible representation than traditional categorical labels.

The system attaches three regression heads to the last hidden layer of BLIP-2's Q-Former, a state-of-the-art vision-language model developed by Salesforce. BLIP-2 first encodes images using a frozen CLIP vision backbone, then maps them through a Q-Former whose outputs are typically decoded into text by a language model. Although BLIP-2 is primarily designed for text-based queries, the Q-Former produces rich visual embeddings that can also be leveraged for regression tasks.

## Dataset

Because no existing dataset is suitable for this task, I collected **30,928 paintings** from WikiArt using a custom scraper. Drawings and prints were excluded. Movement and genre metadata were extracted directly from WikiArt, but stylistic attributes required original expert annotation through the active learning interface.

## Methodology

### Preliminary Training

To prepare the model for expert-annotated fine-tuning, preliminary training was conducted on the WikiArt dataset using hard labels for the movement and genre regression heads. The style (form) head was excluded at this stage, as no large-scale visual style annotation resource is readily available. The BLIP-2 backbone was fine-tuned with the Q-Former parameters trainable while keeping the CLIP vision encoder frozen.

Optimization used AdamW (learning rate = 5×10⁻⁶, weight decay = 0.01) in combination with a ReduceLROnPlateau scheduler (factor = 0.5, patience = 3). Validation loss decreased consistently from 0.229 at epoch 1 to a minimum of 0.067 at epoch 9, followed by small fluctuations (0.0676–0.0707) through epoch 13, indicating convergence and stable generalization capacity.

### Active Learning Loop with Expert Adjustment

To support expert-in-the-loop model refinement, a full-stack application was developed consisting of a FastAPI backend and a React-based frontend. The model, deployed on the backend, generates predictions for paintings and transmits both image data and predicted embedding scores to the user interface. The expert may then adjust these predictions directly; these adjustments are stored and transmitted back to the backend, where they are immediately used for backpropagation-based fine-tuning.

To minimize latency, the system incorporates a caching mechanism with both a cache lock and a model lock, ensuring rapid prediction retrieval in the frontend while preventing training-inference collisions.

### Complementary Contrastive Learning

In addition to single-image annotation, a contrastive learning component is implemented. The expert is presented with pairs of paintings and asked to rate their stylistic similarity in terms of movement, genre, and formal characteristics. These similarity ratings are then compared with the cosine similarity of the model-predicted embedding vectors for the same pairs.

A contrastive loss term encourages the embedding space to align with art-historically meaningful similarity relationships, producing clusters of semantically related artworks and enabling more interpretable visualizations and similarity-based retrieval than regression alone.

## Extensions

### Search Engine Implementation

The embedding system will be deployed in an interactive search interface that enables users to locate artworks using stylistic and thematic criteria. This provides a tangible demonstration of the model's interpretability and practical value.

Users can adjust sliders corresponding to embedding dimensions (e.g., "Romantic ≥ 0.7," "Dynamic ≤ 0.3") to find paintings that match their desired stylistic criteria. The system uses a two-stage retrieval process: pre-filtering based on existing metadata (artist, period, movement) to restrict the candidate pool, followed by fine-grained ranking using the trained model's embedding vectors. Artworks are ranked by cosine similarity in embedding space to the user-specified attribute targets.

### Cluster Analysis for Art Historical Research

Beyond predictive performance, the project aims to assess and interpret the learned embedding space as a source of new art-historical insight. Clustering purity is evaluated through k-means clustering with silhouette scores and adjusted Rand index. Inter-cluster and intra-cluster distances are computed to quantify stylistic relationships and cohesion. Visualizations using t-SNE or UMAP projections examine cluster boundaries, transitional works, and stylistic gradients.

For example, if Romantic paintings cluster more closely to Impressionist works (mean pairwise distance ≈ 0.3) than to Neoclassical works (mean pairwise distance ≈ 0.7), this provides numerical support for art-historical narratives regarding the transition from Romantic expressiveness to Impressionist perceptual immediacy.

### Focus- and Responsibility-Guided Adversarial Image Generation

This component explores ways to add more perturbations to an adversarial image without making them more noticeable by "hiding" perturbations outside of the focal point of the painting. This approach is particularly effective for historical paintings, as art students have been taught for centuries to maximize attention drawn towards focal points through composition, lighting, line, and negative spaces.

The approach combines a model-derived responsibility map (computed from gradient magnitudes) with an expert-drawn focal mask, using a smoothly varying (cosine-eased) spatial weighting to produce a per-pixel perturbation budget. The responsibility map highlights which pixels the model relies on most when computing embeddings, while the focal mask indicates regions that should remain visually recognizable.

The aim is to compare adversarial examples generated under the same perturbation budget using standard FGSM and PGD methods, and to evaluate whether the proposed approach produces perturbations that are less perceptible to human observers while still effectively fooling the model.


## License

This project is for research and academic use only.  

## Contact

Aaron Wang -  aaron-wang@uiowa.edu
