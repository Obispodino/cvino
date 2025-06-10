Here is a comprehensive project plan for your data science capstone, structured around the 9-step framework you provided.

These are internal notes to be deleted at the end, it works as a guide.

To check out the data set, please visit: https://github.com/rogerioxavier/X-Wines
To download it https://drive.google.com/drive/folders/1LqguJNV-aKh1PuWMVx5ELA61LPfGfuu_?usp=drive_link
---

### **Step 1: Define Your Goal**

This step clarifies the core purpose of your project, which will guide all subsequent decisions.

*   **Why am I making this project?**
    This is a data science capstone project. The primary goal is to demonstrate a complete, end-to-end skill set: data acquisition (via user input/image), data modeling (k-NN), business logic application (food pairing rules), and user-facing deployment. It serves as a key portfolio piece to showcase practical, applied ML skills.

*   **Who is this project for?**
    The immediate user is a wine enthusiast who wants quick, personalized recommendations and food pairing ideas without deep wine knowledge. The secondary (and equally important) audience is your academic assessors and future employers, who are evaluating your technical abilities and problem-solving approach.

*   **What will make it valuable?**
    Its value lies in its integration. While other projects might recommend a wine or classify it, this project provides a complete user journey: from identifying a wine they already like (input) to discovering new, similar wines (recommendation) and learning how to enjoy them (food pairing). The combination of these three features in a single, easy-to-use tool is the unique value proposition.

### **Step 2: Write User Stories**

These are simple, non-technical descriptions of what a user can do. This ensures you build what the user actually needs.

1.  As a user, I can input the name of a wine to get information about it.
2.  As a user, I can upload a picture of a wine label to identify the wine.
3.  As a user, I should receive feedback if my uploaded image is unclear or cannot be read.
4.  As a user, I should receive feedback if the wine I searched for is not in the database.
5.  As a user, once a wine is identified, I can view its key details (e.g., country, rating, type).
6.  As a user, I can click a button to get recommendations for wines similar to my input wine.
7.  As a user, I can see a list of 3-5 recommended wines.
8.  As a user, I can see the key details for each recommended wine.
9.  As a user, I can understand *why* a wine was recommended (e.g., "similar taste profile").
10. As a user, I can view food pairing suggestions for my original wine.
11. As a user, I can view food pairing suggestions for each of the recommended wines.
12. As a user, the food pairings should be simple and easy to understand (e.g., "Pairs well with steak, grilled chicken").
13. As a user, I can navigate the application easily on my phone or computer.
14. As a user, the results should load in a reasonable amount of time.
15. As a user, I want the interface to be clean and uncluttered.
16. As a user, I can start a new search at any point in the process.

### **Step 3: Define Your Data Models**

This is an abstract blueprint of your data, independent of technology.

*   **Wine Model:** This is your core model, directly based on the X-Wines dataset.
    *   **Fields:** `WineID`, `WineName`, `Country`, `Region`, `Winery`, `Rating`, `NumberOfRatings`, `Price`, `Year`, and the other 8 attributes from the dataset. This is the central object of your application.

*   **Food Pairing Rule Model:** This model connects wine characteristics to food. This avoids hard-coding pairings and makes the system more scalable.
    *   **Fields:** `RuleID`, `WineAttribute` (e.g., 'Type'), `AttributeValue` (e.g., 'Bold Red'), `SuggestedFood` (e.g., 'Red Meat, Hard Cheese').
    *   **Example:** A rule could be: IF `Wine.Type` is 'Bold Red', THEN suggest 'Steak', 'Lamb', 'Aged Cheddar'.

*   **No User Model (for now):** For a capstone project, you likely don't need user accounts. This simplifies the project significantly by avoiding the need for authentication, passwords, and user data storage.

### **Step 4: Nail an MVP (Minimum Viable Product)**

This is the simplest, core version of your project that delivers value. It's what you'll build first.

*   **Input:** Text-based search for a wine **only**. Image recognition is complex and can be added later as a stretch goal.
*   **Core Logic:** The k-NN model that takes a `WineID` and returns the IDs of the top 3 most similar wines from the X-Wines dataset.
*   **Output:** A single results page that displays:
    1.  The user's input wine.
    2.  A list of the 3 recommended wines.
    3.  A simple, rule-based food pairing suggestion for all 4 wines (the input + 3 recommendations).

### **Step 5: Draw a Simple Wireframe**

This is a basic sketch of your app's flow.

1.  **Home Screen:**
    *   A large title: "Find Your Next Favorite Wine"
    *   A single text input box: "Enter a wine name..."
    *   One button: "Recommend"

2.  **Results Screen:**
    *   **Section 1: "Your Wine"**
        *   Card displaying the details of the wine the user entered.
        *   A list of food pairings for this wine.
    *   **Section 2: "Similar Wines You'll Love"**
        *   Three distinct cards, one for each recommended wine.
        *   Each card shows the wine's name, key details (country, rating), and its specific food pairing suggestions.

This simple two-screen flow achieves the entire MVP goal.

### **Step 6: Understand the Future of Your Project**

*   **Primary Purpose:** A portfolio piece. This means the code should be clean, well-commented, and organized in a public Git repository with a clear `README.md` file explaining how to run it.
*   **Scalability:** Not a concern for the MVP. You don't need to plan for millions of users. The goal is a flawless demo, not a production-ready system. This allows you to choose simpler, faster technologies.
*   **Post-Capstone Life:** After submission, it will serve as a live demo of your skills. Therefore, it needs to be deployed on a reliable (and preferably free/cheap) platform.

### **Step 7: Drill Into Specific Components (Architecture)**

This defines the high-level technical structure.

*   **Backend API:** This is the brain. It will be a Python-based web server.
    *   It will load the X-Wines dataset (e.g., from a CSV or a simple database).
    *   It will load the pre-trained k-NN model.
    *   It will contain the food pairing logic.
    *   It will expose simple API endpoints like `/recommend?wine_name=...` that the frontend can call.
*   **Frontend Interface:** This is the face. It will be a simple web page.
    *   It will have the HTML structure from your wireframe.
    *   It will use JavaScript to take the user's input, send it to your backend API, and then display the results dynamically on the page without a page refresh.
*   **Model File:** The trained k-NN model will be saved to a file (using `pickle` or `joblib`) that the backend API loads on startup.

### **Step 8: Pick Your Stack**

Choose the simplest tools that get the job done, leveraging your data science skills.

*   **Language:** **Python**. It's the language of data science and perfect for your backend.
*   **Backend Framework:** **Flask** or **FastAPI**. Both are lightweight, easy to learn, and perfect for building the simple API you need.
*   **Data Science Libraries:** **Pandas** (for data manipulation), **Scikit-learn** (for the k-NN model).
*   **Database:** **SQLite**. It's a file-based database built into Python. You don't need a separate database server, making setup incredibly simple. You can pre-load the X-Wines CSV into a SQLite file.
*   **Frontend:** **HTML, CSS, and vanilla JavaScript**. No need for a complex framework like React or Vue. A simple `fetch` call in JavaScript is all you need to communicate with your backend.
*   **Deployment:** **PythonAnywhere** or **Heroku's new eco/mini plans**. These platforms are designed for deploying simple Python/Flask apps and are well-documented.

### **Step 9: Overall Development Process**

This is your step-by-step coding plan.

1.  **Project Setup:** Create a Git repository. Set up a Python virtual environment. Create your folder structure (`/app`, `/data`, `/notebooks`).
2.  **Data & Model Prototyping:** In a Jupyter Notebook, load the X-Wines data with Pandas. Clean it, explore it, and build/train your k-NN model. Save the final, trained model to a `.pkl` or `.joblib` file.
3.  **Backend First (API):** Create your Flask/FastAPI application. Write the code to load the dataset and the saved model. Create the `/recommend` endpoint. Test it thoroughly using your browser or a tool like Postman to ensure it returns the correct data in JSON format.
4.  **Build the Frontend:** Write the HTML and CSS for your two screens. Then, write the JavaScript to handle the button click, call your backend API, receive the JSON response, and populate the HTML with the results.
5.  **Deployment:** Deploy your working MVP to PythonAnywhere or another host early. This ensures you know the deployment process works and avoids last-minute issues.
6.  **Iteration (Post-MVP):** Once the core MVP is deployed and working, you can start on stretch goals, beginning with the most important one: **image recognition**. You can add a new endpoint and frontend feature for this, iterating on your solid foundation.
