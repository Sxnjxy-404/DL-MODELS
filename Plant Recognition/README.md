 🌱 Professional Potato Disease AI Diagnostics

This project is an advanced AI-powered system designed to accurately diagnose potato plant diseases from images. Developed as a Streamlit web application, it provides a user-friendly interface for farmers, agronomists, and researchers to get real-time diagnostic insights.

#### Key Features

  * **AI-Powered Diagnosis:** Utilizes a pre-trained TensorFlow model (`potatoes.h5`) to classify potato diseases.
  * **Disease Information:** Provides detailed information on three classes: "Early Blight," "Late Blight," and "Healthy".
  * **Detailed Insights:** For each disease, the application displays scientific names, severity levels, common symptoms, potential causes, recommended treatments, and prevention strategies.
  * **Confidence Score:** The diagnostic engine provides a calibrated confidence score for each prediction, categorized as "Very High," "High," "Medium," "Low," or "Very Low".
  * **Advanced Controls:** The application includes a sidebar with advanced controls for adjusting the confidence threshold and applying various image enhancements such as brightness, contrast, and sharpness.
  * **Responsive UI:** The web interface is professionally styled with custom CSS to provide a modern, responsive, and intuitive user experience.

#### Technology Stack

  * **Python:** The core programming language for the application.
  * **Streamlit:** Used for building and deploying the interactive web interface.
  * **TensorFlow & Keras:** The foundation for the deep learning model used for image classification.
  * **Numpy:** Utilized for efficient numerical operations and array manipulation in the image processing pipeline.
  * **Pillow (PIL):** Used for advanced image processing and enhancement.
  * **OpenCV (cv2):** Implemented for image feature extraction, including sharpness analysis.
  * **Jupyter Notebook:** An untitled notebook file indicates the development and testing environment.

#### Installation & Setup

1.  **Clone the Repository:**
    `git clone <repository_url>`

2.  **Install Dependencies:**
    The following libraries are required to run the application. It is recommended to use a virtual environment.

    ```
    !pip install tensorflow
    !pip install matplotlib
    ```

    The `app.py` script also imports other libraries like `streamlit`, `numpy`, `pillow`, and `opencv-python`. Ensure these are installed in your environment.

3.  **Run the Application:**
    The application can be launched using Streamlit.
    `streamlit run app.py`

#### The AI Model

The application uses a trained model named `potatoes.h5`. This model was likely trained on a dataset from the "PlantVillage" library. The model is capable of classifying images into one of three distinct categories:

  * `Potato___Early_blight`
  * `Potato___Late_blight`
  * `Potato___healthy`

The model has an expected input resolution of 256x256 pixels with 3 color channels, and a reported accuracy of 98.5%.
