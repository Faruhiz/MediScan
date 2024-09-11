# MediScan
## Project Overview
MediScan is a medical imaging project aimed at classifying and analyzing medical images using a deep learning model. The project will leverage pre-trained Convolutional Neural Networks (CNNs) and fine-tune them using transfer learning.

## Technologies
- Database:  Local storage for images; MySQL or PostgreSQL for metadata.

- Model: Start with pre-trained CNN (Resnet) then fine-tune using transfer learning.

## Protocols

### 1. **RESTful APIs**
   - **Endpoints:**
     - `/upload (label, unlabel)`: For uploading new images data. **
     - `/upload(zip)`: Upload image datasets in zip format. **
     - `/get_image_list()`: Retrieve a list of all uploaded images. **
     - `/set_image(id)`: Set an image as labeled or for model training.
     - `/modify_image(id, label)` : Modify the label or properties of an image.
     
     - `/getImage/{id}`: Get image by id.
     - `/createProject`: Initialize the project.

     - `/evaluate`(): Evaluate the current model performance.
     - `/train(model)`: Train the model using the labeled dataset.
     - `/deploy(model_id)` : Deploy a specific version of the model.
     - `/grade(image)`: Predict using the model to classify an image.
     - `/approve(id, status)`: Approve or reject image classifications or modifications.

     - `/feedback`: For sending feedback from the doctor (labeling incorrect results).
     - `/history`: To retrieve past scans and their results.

   - **Data Format:** JSON or CSV
   - **Image Type:** JPG or Yaml 
   - **Image Status:** Modified, Approved

### 2. **MySQL Or MongoDB (?)**
   ### 2.1 **MySQL**
   - **Advantages:**
     - Structured Data: Ideal for structured schema with relationships between entities.
     - ACID Compliance: Ensures data integrity and supports complex transactions.
     - Widely Used: Good support and documentation.

   - **Disadvantages:**
     - Schema Rigidity: Changes in schema require migrations and can be complex.
     - Scalability: Can face challenges with horizontal scaling.
     
   ### 2.2 **MongoDB**
   - **Advantages:**
      - Flexible Schema: Ideal for handling varied data structures and evolving schemas.
      - Scalability: Easier to scale horizontally.
      - Document-Based: Stores data in a flexible JSON-like format (BSON), which can be useful for storing metadata and annotations.
   - **Disadvantages:**
   - ACID Transactions: Less robust compared to MySQL, although improvements have been made.
   - Complex Queries: Might be less efficient for complex joins and transactions.

   **Recommendation for MediScan:**
   - MySQL is better suited if your project requires complex transactions and a structured schema with clear relationships.
   - MongoDB can be a better option if you need flexibility in schema design and are dealing with evolving or varied data structures.

   **Storing Images in the Database:**
      - For storing images, it is generally advised not to store images directly in the database for large datasets (like medical images) because:
      - **BLOB (Binary Large Objects) in relational databases can lead to performance issues when querying or handling large images.**
      - **Best Practice: Store the image files in an external storage (e.g., local filesystem, cloud storage ), and keep references (e.g., file paths, URLs) in the database.**
         - Store images in a structured directory format (e.g., /images/patient_x/scan_y.jpg)
         - Save metadata in the database (Include image metadata (e.g., resolution, modality) in structured columns).      

### 3. Security Protocol for Storing Images and Metadata:**
   1. Data Encryption:
      - In Transit: Use SSL/TLS for encrypting data between clients and your database server.
      - At Rest: Encrypt the storage where the images are stored using file encryption (for local storage) or cloud encryption features (for services like S3).
      - Hash the password.
   2. Access Control:
      - Use OAuth2 or JWT tokens for API access.
      - Implement Role-Based Access Control (RBAC) to define who can upload, modify, or retrieve images (doctors, admins).
   3. Backup and Recovery:
      - Implement a backup system (daily/weekly) for both the database and images.
      - Ensure backups are also encrypted.
   4. Audit Logs:
      - Enable logging for sensitive operations such as image uploads, modifications, and deletions.
   5. Data Integrity:
      - Use checksums to ensure image files are not corrupted during transfer or storage.

### 4. **AI Framework - Using MLflow for MediScan**
Start with MLflow for ease of use and integration, consider other tools as needed for scaling:
   - Tracking experiments: You can track model parameters, metrics, and outputs (scan results, predictions).
   - Model versioning: Keep track of different versions of your model (ResNet pre-trained and fine-tuned versions).
   - Model deployment: MLflow can help you deploy models as APIs or web services for easy integration with your system.

**Alternative AI Tools:**
   - TensorFlow Serving: Specialized for serving TensorFlow models in production.
   - Kubeflow: A Kubernetes-based solution for machine learning workflows if you're scaling and using cloud infrastructure.

## Datasets

### 1. **Image Data**
   - **Format:** 
     - Formats like **JPEG**, **PNG**, or **TIFF**
   - **Resolution:** 
     - High resolution as required for medical imaging.
   - **Storage:** 
     - Store the images in a structured directory format.

```
    /dataset/
    ├── train/
    │   ├── class_1/
    │   └── class_2/
    ├── val/
    │   ├── class_1/
    │   └── class_2/
    └── test/
        ├── class_1/
        └── class_2/
```

### 2. **Labels and Annotations**
   - **Format:** 
     - Use **CSV** or **JSON**
     
     **Example**
       ```csv
       image_id, label
       img001.jpg, 0
       img002.jpg, 1
       ```

Classification task:

```
/dataset/
├── train/
│   ├── class_1/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── class_2/
│       ├── img003.jpg
│       └── img004.jpg
├── val/
│   ├── class_1/
│   └── class_2/
├── test/
│   ├── class_1/
│   └── class_2/
└── labels.csv
```

### **Example Dataset Structure:**

### 1. **Database Structure**
   1. **Image Data**
      - Image ID: Unique identifier (e.g., img001)
      - File Name: Name of the image file (e.g., img001.jpg)
      - File Path/URL: Location of the image (e.g., /path/to/img001.jpg)
      - Upload Date: Date and time of upload (e.g. 2024-09-10 12:00:00)
      - Image Format: File format (e.g., JPEG, PNG, TIFF)
      - Resolution: Image resolution (Should be original image resolution)
      - Size: Image size in bytes (e.g., 2MB)
      - Image Modality: Type of imaging (e.g., X-ray, CT scan)
      - Label Status: Whether the image is labeled (e.g., labeled, unlabeled)
      - Approval Status: Approval status (e.g., pending, approved, modified)
      - Doctor/Annotator ID: ID of the person who labeled the image (e.g. dr001)

   2. **Label and Annotation Data**
      - label_id
      - Image ID: Foreign key to the images table
      - Label: Classification label (e.g., 0 for no disease, 1 for disease)
      - Confidence Score: Confidence of the model prediction (e.g., 0.85)
      - Annotation Date: Date and time the label was provided (e.g. 2024-09-10 12:30:00)
      - Doctor/Annotator ID: ID of the annotator (e.g. dr001)

   3. **Metadata for Each Image**
      - metadata_id	
      - Image ID: Foreign key to the images table
      - Pixel Spacing: Spacing between pixels (e.g., 0.2mm)
      - Institution: Institution where the image was captured (e.g. General Hospital)
      - Capture Date: Date and time the image was captured (e.g. 2024-08-25 09:45:00)

   4. **Model Information**
      - Model ID: Unique identifier (e.g., model001)
      - Model Name: Name of the model (e.g., ResNet_v1)
      - Version: Version of the model (e.g., v1.0)
      - Training Date: Date and time the model was last trained (e.g. 2024-09-10 15:00:00)
      - Accuracy: Accuracy of the model (e.g., 0.92)
      - Labeled Dataset: Dataset used to train the model (e.g. labeled_xray_images)

   5. Model Training History**
      - training_id
      - Model ID: Foreign key to the models table
      - Dataset ID: The dataset used for training (e.g. dataset001)
      - Training Start Date: Start date of the training process (e.g. 2024-09-01 08:00:00)
      - Training End Date: End date of the training process (e.g. 2024-09-01 18:00:00)
      - Performance Metrics: Additional metrics such as precision, recall, F1 score (e.g. {"precision": 0.92, "recall": 0.91, "f1_score": 0.91})
   
