# Attribution and source terms

Numerical public job-shop fixtures follow Google OR-Tools public examples, including FT06 (Google LLC example code, Apache License 2.0): https://github.com/google/or-tools/blob/stable/examples/python/jobshop_ft06_sat.py . The solver implementation in this project is independently written against the OR-Tools API.

Technical PDF narratives: U.S. Department of Energy, Determine the Cost of Compressed Air for Your Plant (DOE/GO-102000-0986) and Minimize Compressed Air Leaks (DOE/GO-102004-1964). Sources and exact downloaded SHA256 values are in results/document_provenance.json. See https://www.energy.gov/web-policies for DOE government-information terms and the exception for third-party contributions. The leak-rate table explicitly reproduced by third-party permission is excluded from extracted chunks. Original PDFs retain their notices, are fetched for local research and are excluded from the portable archive. No DOE endorsement is implied. Source text is normalized and chunked; quoted page numbers are physical 1-based PDF pages.

The four local demonstration procedures and production records are authored synthetic fixtures, not plant manuals or real industrial data.

MiniLM: sentence-transformers/all-MiniLM-L6-v2, Apache-2.0. FLAN-T5: google/flan-t5-small, Apache-2.0. The download script pins revisions and records checksums, preserving model cards. Model weights are excluded from Git and the portable archive; download from the original repositories. No models were fine-tuned in this project.
