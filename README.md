# MediScan
## Technologies
- Database: Amazon S3 (or any equivalent eg. Google Cloud Storage) for storing images + NoSQL database for metadata.

- Model: Start with pre-trained CNN (Resnet) then fine-tune using transfer learning.

## Protocols

### 1. **RESTful APIs**
   - **Endpoints:**
     - `/scan`: For uploading and processing new images.
     - `/feedback`: For sending feedback from the doctor (labeling incorrect results).
     - `/history`: To retrieve past scans and their results.

   - **Data Format:** JSON

### 2. **GraphQL**
   - Frontend needs flexible querying, GraphQL can be considered.

   - **Advantages:**
     - Allows the frontend to request exactly the data it needs.
     - Reduces over-fetching or under-fetching of data.

### 3. **Security Protocols**
   - **Authentication:** OAuth2 or JWT
   - **Encryption**
   - **Access Control**

### 6. **File Upload Protocols**
   - use `multipart/form-data` in RESTful APIs to upload images.
   - Ensure that large files are handled properly, perhaps with chunked uploads if necessary.

### 7. **Error Handling**
   - Implement error handling for all APIs.

### 8. **Logging and Monitoring**
   - Implement logging for API requests and responses, especially for errors.
   - Consider using monitoring tools to track API performance and health.

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
  
### 3. **Metadata**
- **Study details** (e.g., imaging modality, date, institution).
- **Image properties** (e.g., resolution, pixel spacing, modality).
   - **Format:**
     - **JSON** or **CSV**

### **Example Dataset Structure:**

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