### Design Diagram: Instagram Feed Ranking Model

Here is a conceptual design diagram that explains the flow of the Instagram feed ranking model as discussed in the video.

**High-Level Architecture**

* **User** -> **Instagram App** -> **Ranking Pipeline** -> **Ranked Feed** -> **User**

**Detailed Ranking Pipeline**

1.  **Candidate Generation:**
    * **Input:** User ID
    * **Process:** Selects an initial large set of ~1000 potential posts for the user's feed.
    * **Output:** A list of candidate posts.

2.  **Ranking:**
    * **Input:** The list of candidate posts from the previous stage.
    * **Process:** A two-tower neural network model is used to predict the probability of user engagement for each candidate post.
        * **Viewer Tower:** Processes user features (e.g., past engagement, demographics).
        * **Post Item Tower:** Processes post features (e.g., content, creator information, historical engagement).
        * The outputs of both towers (embeddings) are combined to generate an engagement score.
    * **Output:** A ranked list of posts based on the engagement score.

3.  **Post-Processing (Re-ranking):**
    * **Input:** The ranked list of posts.
    * **Process:** Applies business logic and heuristics to the ranked list to ensure:
        * **Diversity:** Avoids showing too many similar posts.
        * **Fairness:** Ensures a fair distribution of content from different creators.
        * **Freshness:** Prioritizes newer content.
    * **Output:** The final, re-ranked list of posts that is displayed to the user.

***

### Main Summaries

Here are the main summaries of the key concepts discussed in the video:

* **Problem and Goals:** The main goal is to design a ranking model for suggested posts on the Instagram feed to improve user engagement, which is measured by Daily Active Users (DAU) and session duration [[02:44:03](http://www.youtube.com/watch?v=7_E4wnZGJKo&t=9843)]. The machine learning objective is to predict individual user engagement, such as views, likes, and comments [[04:40:40](http://www.youtube.com/watch?v=7_E4wnZGJKo&t=16840)].

* **System Requirements:** The system must be scalable to handle Instagram's large user base (estimated at 500 million daily active users [[07:11:04](http://www.youtube.com/watch?v=7_E4wnZGJKo&t=25864)]) and be highly available. It also requires robust ML Ops tooling for monitoring and debugging [[09:26:07](http://www.youtube.com/watch?v=7_E4wnZGJKo&t=33967)].

* **Three-Phase Ranking Pipeline:** The ranking process is broken down into three main stages [[11:48:48](http://www.youtube.com/watch?v=7_E4wnZGJKo&t=42528)]:
    1.  **Candidate Generation:** A large set of potential posts is generated [[14:21:00](http://www.youtube.com/watch?v=7_E4wnZGJKo&t=51660)].
    2.  **Ranking:** The generated candidates are sorted based on their predicted engagement probability [[12:09:00](http://www.youtube.com/watch?v=7_E4wnZGJKo&t=43740)].
    3.  **Post-Processing:** The ranked list is re-shuffled to incorporate factors like fairness, diversity, and content freshness [[12:48:00](http://www.youtube.com/watch?v=7_E4wnZGJKo&t=46080)].

* **Features and Data:** The model uses various features for ranking:
    * **Viewer Features:** User's past engagement and activity [[15:51:52](http://www.youtube.com/watch?v=7_E4wnZGJKo&t=57112)].
    * **Post Features:** Information about the post, its creator, and its historical engagement [[17:05:01](http://www.youtube.com/watch?v=7_E4wnZGJKo&t=61501)].
    * **Embeddings:** Vector representations of the video, audio, and text content of the posts [[17:38:00](http://www.youtube.com/watch?v=7_E4wnZGJKo&t=63480)].
    * **Labels:** User interactions (view, like, comment) are used as labels for training the model [[20:21:00](http://www.youtube.com/watch?v=7_E4wnZGJKo&t=73260)].

* **Model Architecture:** The video proposes a **two-tower network** [26:19:00], which is a scalable deep learning model. It consists of two separate neural networks (towers): one for the user (viewer) and one for the post (item). The outputs of these towers are combined to predict the probability of engagement [34:59:00].

* **Evaluation and Deployment:** The model is evaluated both offline and online:
    * **Offline Evaluation:** Metrics like AUC are used to assess the model's performance on a validation dataset [38:57:00].
    * **A/B Testing:** The new model is tested on a small fraction of users in a live production environment to compare its performance against the existing system [42:56:00].
    * **Safeguard Metrics:** It's important to monitor for any negative side effects, such as an increase in users reporting or blocking content [44:18:00].

* **Cold Start Problem:** For new users with no interaction history, a common approach is to show them popular posts to gather initial data on their preferences [47:24:00].

***

### QA Pairs

Here are some question-and-answer pairs based on the video content:

* **Q: What is the main business objective of the Instagram feed ranking model?**
    * A: The primary business objective is to improve user engagement, which in turn contributes to increasing Daily Active Users (DAU) and the average session duration [[02:44:03](http://www.youtube.com/watch?v=7_E4wnZGJKo&t=9843)].

* **Q: What are the three main stages of the ranking pipeline?**
    * A: The three main stages are Candidate Generation, Ranking, and Post-Processing (or re-ranking) [[11:48:48](http://www.youtube.com/watch?v=7_E4wnZGJKo&t=42528)].

* **Q: What kind of model architecture is proposed for the ranking model?**
    * A: A two-tower neural network is proposed. This model has two separate "towers," one for processing user features and one for processing post features, which makes it highly scalable [33:18:00].

* **Q: What are some of the features used to rank a post?**
    * A: The model uses viewer features (like past engagement), post features (like creator information and historical engagement on the post), and embeddings generated from the video, audio, and text of the post [[15:51:52](http://www.youtube.com/watch?v=7_E4wnZGJKo&t=57112), [17:38:00](http://www.youtube.com/watch?v=7_E4wnZGJKo&t=63480)].

* **Q: How is the model's performance evaluated before it's released to all users?**
    * A: The model is first evaluated offline using metrics like AUC on a validation dataset. Then, it's tested in a live environment using A/B testing on a small percentage of users to compare its performance against the current system [38:57:00, 42:56:00].

* **Q: How does the system handle new users who have no interaction history (the "cold start" problem)?**
    * A: For new users, a common approach is to initially recommend popular posts to them and then observe their interactions to start learning their preferences [47:24:00].
