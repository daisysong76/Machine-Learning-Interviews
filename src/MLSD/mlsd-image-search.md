# Image Search System (Pinterest)

### 1. Problem Formulation
* Clarifying questions
    - What is the primary (business) objective of the visual search system?
    - What are the specific use cases and scenarios where it will be applied?
    - What are the system requirements (such as response time, accuracy, scalability, and integration with existing systems or platforms)?
    - How will users interact with the system? (click, like, share, etc)? Click only
    - What types of visual content will the system search through (images, videos, etc.)? Images only 
    - Are there any specific industries or domains where this system will be deployed (e.g., fashion, e-commerce, art, industrial inspection)?
    - What is the expected scale of the system in terms of data and user interactions?
    - Personalized? not required 
    - Can we use metadata? In general yes, here let's not. 
    - Can we assume the platform provides images which are safe? Yes
* Use case(s) and business goal
  * Use case: allowing users to search for visually similar items, given a query image by the user 
  * business goal: enhance user experience, increase click through rate, conversion rates, etc (depends on use case)
* Requirements
  * response time, accuracy, scalability (billions of images)
* Constraints
  * budget limitations, hardware limitations, or legal and privacy constraints
* Data: sources and availability
  * sources of visual data: user-generated, product catalogs, or public image databases?
  * Available? 
* Assumptions
* ML formulation: 
  * ML Objective: retrieve images that are similar to query image in terms of visual content 
  * ML I/O: I: a query image, and O: a ranked list of most similar images to the query image 
  * ML category: Ranking problem (rank a collection of items based on their relevance to a query)

### 2. Metrics  
* Offline metrics 
  * MRR 
  * Recall@k 
  * Precision@k 
  * mAP 
  * nDCG 
* Online metrics 
  * CTR 
  * Time spent on images 
1. MRR (Mean Reciprocal Rank)
Use: Measures how high the first relevant item is ranked.
Pros: Simple and quick to compute.
Cons: Ignores all other relevant items; poor for assessing full ranking quality.
Trade-off: Good for tasks where only the first relevant item matters (e.g., QA); not suitable for ranking tasks with multiple relevant items.
2. Recall@k
Use: Measures how many of the relevant items appear in the top k predictions.
Pros: Captures coverage of relevant items.
Cons: Biased in datasets with many relevant items; doesn’t care about where in the top-k the items appear.
Trade-off: May underrepresent model performance in domains with high cardinality (e.g., image-to-image retrieval).
3. Precision@k
Use: Proportion of relevant items in the top k predictions.
Pros: Easy to understand; good for evaluating top-k quality.
Cons: Ignores ordering within the top k; insensitive to the position of items.
Trade-off: Useful for strict precision targets; lacks depth for ranking-sensitive applications.
4. mAP (Mean Average Precision)
Use: Averages precision across all relevant items in a list.
Pros: Better captures ranking quality across the whole list.
Cons: Assumes binary relevance (relevant or not); not suitable for graded relevance.
Trade-off: Robust for tasks with binary labels; less ideal for tasks with graded similarity (like image scores 0–5).
5. nDCG (Normalized Discounted Cumulative Gain)
Use: Measures ranking quality considering both relevance and position.
Pros: Handles graded relevance well; best offline metric for evaluating ranking fidelity.
Cons: Requires ideal relevance labels (which can be hard to define).
Trade-off: Most accurate but computationally heavier; ideal for tasks with ordinal or continuous relevance (e.g., image similarity 0–5).
🌐 Online Metrics
These are collected via real user interactions on the live platform.
1. CTR (Click-Through Rate)
Use: Measures how often users click on items shown.
Pros: Direct signal of user interest; easy to measure.
Cons: Can be influenced by item position, thumbnail attractiveness, etc.
Trade-off: Simple but may not reflect deep engagement or relevance.
2. Time Spent on Images
Use: Measures how long users view/interact with an image.
Pros: Captures depth of engagement.
Cons: Noisy signal (e.g., user distracted); doesn’t always imply satisfaction.
Trade-off: Complements CTR well; useful for detecting "attractive but unhelpful" content.
### 3. Architectural Components  
* High level architecture 
  * Representation learning: 
    * transform input data into representations (embeddings) - similar images are close in their embedding space 
    * use distance between embeddings as a similarity measure between images 

### 4. Data Collection and Preparation
* Data Sources
  * User profile
  * Images 
    * image file
    * metadata
  *  User-image interactions: impressions, clicks: 
  * Context 
* Data storage
* ML Data types
* Labelling

### 5. Feature Engineering
* Feature selection 
  * User profile : User_id, username, age, gender, location (city, country), lang, timezone
  * Image metadata: ID, user ID, tags, upload date, ... 
  * User-image interactions: impressions, clicks: 
    * user id, Query img id, returned img id, interaction type (click, impression), time, location
* Feature representation 
  * Representation learning (embedding)
* Feature preprocessing 
  * common feature preprocessing for images: 
    * Resize (e.g. 224x224), Scale (0-1), normalize (mean 0, var 1), color mode (RGB, CMYK) 

### 6. Model Development and Offline Evaluation
* Model selection 
  * we choose NN because of 
    * unstructured data (images, text) -> NN good at it 
    * embeddings needed 
  * Architecture type: 
    * CNN based e.g. ResNet 
    * Transformer based (ViT)
    * Example: Image -> Convolutional layers -> FC layers -> embedding vector  
* Model Training 
  * contrastive learning -> used for image representation learning 
    * train to distinguish similar and dissimilar items (images)
* Dataset 
  * each data point: query img, positive sample (similar to q), n - 1 neg samples (dissimilar)
    * query img : randomly choose 
    * neg samples: randomly choose 
    * positive samples: human judge, interactions (e.g. click) as a proxy, artificial image generated from q (self supervision)
      * human: expensive, time consuming 
      * interactions: noisy and sparse 
      * artificial: augment (e.g. rotate) and use as a positive sample (similar to simCLR or MoCo) - data distribution differs in reality 
* Loss Function: contrastive loss 
  * contrastive loss: 
    * works on pairs (Eq, Ei)
    * calculate distance: b/w pairs -> softmax -> cross entropy <- Labels 
* Model eval and HP tuning 
* Iterations 
  
### 7. Prediction Service
* Prediction pipeline 

  * Embedding generation service 
    * image -> preprocess -> embedding gen (ML model) -> img embedding 
  * NN search service 
    * retrieve the most similar images from embedding space 
      * Exact: O(N.D)
      * Approximate(ANN) - sublinear e.g. O(D.logN)
        * Tree based ANN (e.g. R-trees, Kd-trees) 
          * partition space into two (or more) at each non-leaf node, 
          * only search the partition for query q 
        * Locality Sensitive Hashing LSH 
          * using hash functions to group points into buckets (close points into same buckets)
        * Clustering based 
    * We use ANN using an existing library like Faiss (Facebook)
  * Re-ranking service 
    * business level logic and policies (e.g. filter inappropriate or private items, deduplicate, etc)
* Indexing pipeline
  * Indexing service: indexes images by their embeddings 
  * keep the table updated for new images 
  * increases memory usage -> use optimization (vector / product quantization)

### 8. Online Testing and Deployment  
* A/B Test 
* Deployment and release 

### 9. Scaling, Monitoring, and Updates 
* Scaling (SW and ML systems)
* Monitoring 
* Updates 

### 10. Other points: 

