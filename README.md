# AI Perspective Crop

Automatic document perspective correction using OpenCV.

This project implements a full pipeline for detecting a document in an image and applying perspective transformation to generate a clean, top-down scanned result — similar to mobile document scanner apps.

---

## 🚀 Features

- Automatic document detection
- GrabCut-based foreground segmentation
- Convex Hull refinement
- Rotated bounding box (minAreaRect)
- Perspective transform
- Clean and minimal final output

---

## 🧠 Core Idea

Instead of relying only on edge detection, the final version uses:

1. **GrabCut segmentation** to isolate the document region  
2. **Convex Hull** to remove concave noise caused by text or shadows  
3. **minAreaRect** to fit the optimal rotated rectangle  
4. **Perspective transform** to generate a flat scanned result  

This makes the detection more stable and robust compared to pure Canny/Hough-based approaches.

