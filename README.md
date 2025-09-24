# Balance Sheet Recognition

## Overview

**Balance Sheet Recognition** is an AI-powered project designed to automatically extract and analyze balance sheets from financial PDF statements. Traditional methods involve printing PDFs, entering data into Excel, and calculating financial ratios—this process is time-consuming and prone to errors.  

This project uses **computer vision** and **NLP techniques** to detect balance sheets, extract key information, and calculate financial ratios efficiently.  

Models used: **YOLOv11** for table/balance sheet detection and **LayoutLMv3** for token-level understanding and labeling.

---

## Features

- Detect balance sheets in PDF documents  
- Extract key-value pairs, years, and company names  
- Annotate text and bounding boxes using Label Studio  
- Data preparation pipeline for LayoutLMv3 training  
- Automatically calculate financial ratios after extraction  
- Optional **data augmentation** using Albumentations for image preprocessing  

---

## Project Architecture

### 1. Data Understanding
- Analyze financial PDF statements to identify text, tables, and balance sheets  
- Focus on balance sheets from the industrial sector  

### 2. Data Preparation
- Convert PDFs to images using `pdf2image`  
- Annotate balance sheets with Label Studio (`key`, `value`, `year`, `name`)  
- Save annotations in JSON format for model training  

### 3. Modeling
- **YOLOv11**: Detect balance sheet tables in images  
- **LayoutLMv3**: Token-level recognition of key fields using OCR and bounding boxes  

### 4. Evaluation
- Compute **precision, recall, F1-score, and accuracy** for token-level predictions  

### 5. Deployment
- Automatically extract balance sheets and calculate financial ratios  

---

## Installation

```bash
git clone https://github.com/Salma5806/balance-sheet-recognition.git
cd balance-sheet-recognition
pip install -r requirements.txt
