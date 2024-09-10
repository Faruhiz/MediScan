# MediScan
## Technologies
- Database: Local Computer Storage for storing images + MySQL,SQL Lite database.

- Model: Start with pre-trained CNN (Resnet) then fine-tune using transfer learning.

## Protocols

### 1. **RESTful APIs**
   - **Endpoints:**
     - `/upload (label, unlabel)`: For uploading new images data.
     - `/upload(zip)`: Upload image datasets in zip format.
     - `/get_image_list()`: Retrieve a list of all uploaded images.
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

   - **Data Format:** JSON
   - **Image Type:** JPG or Yaml 
   - **Image Status:** Modified, Approved

### 2. **MySQL Or PostgreSQL Or MongoDB (?)**
   - Frontend might needs flexible querying, GraphQL can be considered.
   ### 2.1 **MySQL**
   - **Advantages:**
     - Ease of Use: MySQL is beginner-friendly and often considered easier to set up and manage, especially for smaller applications.
     - Widely Adopted: It has a large community and ecosystem, with excellent compatibility with web technologies like PHP and WordPress.
     - Speed: For read-heavy operations and simple queries, MySQL tends to be faster due to its architecture.
     - Replication: Simple and effective master-slave replication options for scaling read performance.
   - **Disadvantages:**
     - Limited Advanced Features: MySQL lacks some advanced features available in PostgreSQL, such as partial indexes, full-text search across multiple languages, and rich data types like JSONB.
     - ACID Compliance: MySQL's default storage engine, MyISAM, is not ACID-compliant. InnoDB is more robust but might require additional tuning.
     - Concurrency Control: PostgreSQL generally handles high-concurrency better, especially for write-heavy workloads.
   ### 2.2 **PostgreSQL **
   - **Advantages:**
      - Advanced Features: PostgreSQL offers powerful features like complex queries, full-text search, JSONB support, and advanced indexing techniques.
      - ACID Compliance: PostgreSQL is fully ACID-compliant by default, ensuring higher reliability for transaction-based applications.
      - Extensibility: You can extend PostgreSQL with custom data types, functions, and even procedural languages like PL/pgSQL, Python, etc.
      - Concurrency: PostgreSQL uses Multi-Version Concurrency Control (MVCC), which allows better performance under high-load scenarios with concurrent transactions.
   - **Disadvantages:**
   - Steeper Learning Curve: PostgreSQL's advanced features come with a steeper learning curve compared to MySQL.
   - Performance: For simple, read-heavy queries, PostgreSQL may be slightly slower than MySQL due to the extra overhead of MVCC.

   **Recommendation for MediScan:**
   - PostgreSQL is better suited if you require advanced data manipulation, ACID-compliance, and support for complex queries involving medical image metadata (e.g., JSONB fields).
   - MySQL can be a better option if your primary focus is simplicity and speed for basic operations.

   **Storing Images in the Database:**
      - For storing images, it is generally advised not to store images directly in the database for large datasets (like medical images) because:
      - **BLOB (Binary Large Objects) in relational databases can lead to performance issues when querying or handling large images.**
      - **Best Practice: Store the image files in an external storage (e.g., local filesystem, cloud storage ), and keep references (e.g., file paths, URLs) in the database.**
         - Store images in a structured directory format (e.g., /images/patient_x/scan_y.jpg)
         - Save metadata in the database (Include image metadata (e.g., resolution, modality) in structured columns).
   
   **Steps to Store Images Using References:**
   or use cloud storage (Amazon S3).
   2. Save metadata in the database:
      - Store the image path or URL in a database field.
      

### 3. Security Protocol for Storing Images and Metadata:**
   1. Data Encryption:
      - In Transit: Use SSL/TLS for encrypting data between clients and your database server.
      - At Rest: Encrypt the storage where the images are stored using file encryption (for local storage) or cloud encryption features (for services like S3).
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
MLflow is a great choice for managing your machine learning lifecycle, particularly for a project like MediScan that requires:
   - Tracking experiments: You can track model parameters, metrics, and outputs (scan results, predictions).
   - Model versioning: Keep track of different versions of your model (ResNet pre-trained and fine-tuned versions).
   - Model deployment: MLflow can help you deploy models as APIs or web services for easy integration with your system.

**Advantages of MLflow:**
   1. Experiment Tracking: Easy to compare different versions of your CNN models (ResNet, fine-tuning stages).
   2. Model Registry: Maintain a registry of models, allowing you to seamlessly deploy or rollback specific versions.
   3. Integration: Works well with Python-based ML libraries (TensorFlow, PyTorch, etc.), which is ideal for a CNN-based model.
   4. Deployment: MLflow simplifies deploying models into production using REST APIs.
**Alternative AI Tools:**
   - TensorFlow Serving: Specialized for serving TensorFlow models in production.
   - Kubeflow: A Kubernetes-based solution for machine learning workflows if you're scaling and using cloud infrastructure.
**Recommendation:**
   - Start with MLflow for tracking, versioning, and deployment of your models. If scaling becomes necessary, consider Kubeflow later.

## Datasets

### 1. **Image Data**
   - **Format:** 
     - Formats like **JPEG**, **PNG**, or **TIFF**
   - **Resolution:** 
     - Should be original resolution, as medical images often require high detail.
   - **Storage:** 
     - Store the images in a structured directory format or use an object storage service like Amazon S3. Example directory structure:

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
     - Use **CSV** or **JSON** files to store labels and annotations.
     - eg.,

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
   
