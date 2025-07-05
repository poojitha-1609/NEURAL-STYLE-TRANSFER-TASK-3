# NEURAL-STYLE-TRANSFER-TASK-3
🗂️ Project Title: Neural Style Transfer – Artistic Image Styling

🎯 INTRODUCTION:
Neural Style Transfer (NST) is a deep learning technique that combines the content of one image with the artistic style of another. For example, it allows turning a photo of a street into a painting in the style of Van Gogh.

In this internship task, we built a fully functional Python-based Neural Style Transfer system that uses pre-trained models from PyTorch’s torchvision.models. The system reads two images — a content image and a style image — and generates a stylized output image.

✨ OBJECTIVE:
To implement a neural style transfer model in Python that:
Accepts a content image (e.g., a photo)
Accepts a style image (e.g., a painting)
Produces a new image preserving the content but adopting the artistic style

The final script outputs a high-quality stylized image, ready for viewing or saving.

📌 PROBLEM STATEMENT:
Manually applying artistic effects to photographs requires skill and effort. With deep learning, we can automate this process. The challenge lies in extracting meaningful content and stylistic elements from two different images and blending them using a loss-optimized image generation process.

🛠 TECHNOLOGIES USED:
      Technology	                                          Purpose
      Python 3.8+	                                    Core programming language
      PyTorch	                                         Deep learning framework
      Torchvision	                                     Pre-trained VGG19 model
      Matplotlib	                                        Displaying images
      PIL (Pillow)	                                Image manipulation and saving

💻 SYSTEM REQURIMENTS:
Python ≥ 3.8
PyTorch with CUDA support (for GPU acceleration)
8 GB RAM or higher
NVIDIA GPU (optional but recommended)
Internet connection (for model download)
.jpg or .png content and style images

🔍 METHODOLOGY:
Steps:
Load and normalize content and style images
Load a pre-trained VGG19 network
Extract feature maps from specific VGG layers
Compute content loss (how different is the generated image from the content)
Compute style loss (difference in Gram matrices between style and generated image)
Minimize a combined loss via gradient descent
Save and display the final output image

🖼 DATASEET / INPUT IMAGES:
This project uses two images:
Content Image: A photo or image you want to preserve (e.g., a face or landscape)
Style Image: An artwork or design pattern whose style is to be applied (e.g., "Starry Night")

Note: You can replace the default content.jpg and style.jpg with any high-resolution image.

🧠 NEURAL STYLE TRANSFER THEORY:
Neural Style Transfer works by separating and recombining the content and style of images using a Convolutional Neural Network (CNN).
Content is extracted from deep layers of VGG19 (e.g., conv4_2)
Style is extracted by calculating Gram matrices from shallow and mid-level layers 
(e.g., conv1_1 to conv5_1)

The loss function used is:
Total Loss = α * Content Loss + β * Style Loss
Where:
α is the weight for content loss
β is the weight for style loss

💡 CODE WALKTHROUGH:
The core of this project is the Python script. Below is a simplified breakdown:
a. Load and preprocess images:

content = load_image("content.jpg")
style = load_image("style.jpg")
b. Load pre-trained VGG19 model:

vgg = models.vgg19(pretrained=True).features.to(device).eval()
c. Extract content and style features:

content_features = get_features(content, vgg, content_layers)
style_features = get_features(style, vgg, style_layers)
d. Create the target image:

target = content.clone().requires_grad_(True)
e. Define loss and optimizer:

optimizer = optim.Adam([target], lr=0.003)
f. Optimize image:

for i in range(1, steps+1):
    ...
    total_loss = content_weight * content_loss + style_weight * style_loss
    ...
g. Save and display output:

final_img = im_convert(target)
final_img.save("styled_output.jpg")
final_img.show()

📷 OUTPUT SAMPLE:
With a content image of a building and a style image of "The Great Wave off Kanagawa", the model produces a building that appears painted in the wave’s unique artistic style.
The output image:
Retains the structure of the original photo
Mimics color palette and brush strokes from the style image

⚠ LIMITATIONS:
    Limitation	                                     Description
  Requires powerful GPU	                      Training is slower on CPU
  Memory intensive	                        High-res images may exceed GPU RAM
  Works best on short images	               Complex images may blend poorly
  Style generalization                 	Works for artistic styles, not realistic ones

🚀 FUTURE SCOPE:
Implement real-time NST with webcam feed
Use more efficient models like Fast Neural Style Transfer
Create a web-based interface using Flask or Streamlit
Allow multiple style blending (e.g., Picasso + Van Gogh)
Export results as video or animation

✅ DELIVERABLES CHECKLIST:
        Deliverable                                          	Status
Script that loads and processes images	                        ✅
Uses pre-trained VGG19 model	                                  ✅
Computes style/content loss	                                    ✅
Optimizes target image	                                        ✅
Saves and displays stylized output	                            ✅
Fully documented and explained code                            	✅
Internship-ready report	                                        ✅

📁 FOLDER STRUCTURE:
neural_style_transfer/
├── content.jpg            # Content image
├── style.jpg              # Style image
├── neural_style_transfer.py
└── styled_output.jpg      # Generated output

🖼 EXAMPLE OUTPUT:
content.jpg: Your photo
style.jpg: Artwork (e.g., Van Gogh)
styled_output.jpg: Artistic rendering of content with style applied

✅ SUMMARY OF DELIVERABLE:
Task	Completed                                               ✅
Neural Style Transfer Model                                 	✅
Uses pre-trained VGG19	                                      ✅
Applies artistic style to photo	                              ✅
Saves and displays final output	                              ✅

        ![image](https://github.com/user-attachments/assets/0f28fe48-fc6c-41ff-abec-6cec19e6da3b)


📜 CONCLUSION:
        This project successfully implements Neural Style Transfer using PyTorch and VGG19, demonstrating how deep learning models can blend two forms of visual data — content and style. The system is flexible, easy to use, and delivers visually appealing results. It's a practical introduction to the power of convolutional neural networks in creative AI applications.
