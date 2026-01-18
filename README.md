## Paint By Numbers kit generation
This Repo contains implementation of Paint By Numbers (PBN) kits from input image.
- It provides a Canvas object, which in initialization performs all necessary processing and returns all taken steps.
- It provides gradio app which adds a UI layer to the app, and makes it more usable


To start the app use 
```
uv run python src\app\gradio_app.py
```

### Canvas object:
Canvas object is the main object of the kit. It takes all the processing steps:
1. Formating the input image for A4 format
2. Performing KMeans clustering (contolled by **N_COLORS**)
3. Removing small facets (controlled by **MIN_FACET_PIXELS_SIZE**)
4. Removing narrow facets (controlled by **NARROW_FACET_THRESHOLD_PX**)
5. Creating the outline image with labeled colors

### ColorPalette object:
ColorPalette allows user to additionally rerender the processed image with adjusted color palette.
After the initial Canvas is created it has the color_palette attribute, where we can change given color label, or color value.

### Sample processed image:
![image-1](https://github.com/user-attachments/assets/8a08ca1a-a078-430b-be17-865a2f58ceeb)
![image-2](https://github.com/user-attachments/assets/a1d5a435-e9e6-40e2-88a8-b6768c751228)
