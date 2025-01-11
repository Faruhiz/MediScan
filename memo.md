## CLI Commands

For authentication (use this first!!)
```bash
gcloud auth application-default login --impersonate-service-account medsight@model-argon-428114-m4.iam.gserviceaccount.com
```

For start local server
```bash
uvicorn main:app --reload
```