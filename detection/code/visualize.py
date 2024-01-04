# Visualizing model results

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor

def draw_box(context, corners, color):
  """
  Draw box according to corners information
  """
  
  y_min, x_min, y_max, x_max = corners
  context.rectangle(xy=[(x_min, y_min), (x_max, y_max)], outline=color, width=2)

def draw_text(image, text, position, color):
  """
  Draw class name and score of a bbox
  """
  
  scale=1.5
  font = ImageFont.load_default()
  text_size = font.getsize(text)
  text_image = Image.new(mode="RGBA", size=text_size, color=(0, 0, 0, 0))
  context = ImageDraw.Draw(text_image)
  context.text(xy=(0, 0), text=text, font=font, fill=color)
  scaled_text = text_image.resize((round(text_image.width * scale), round(text_image.height * scale)))
  position = (round(position[0]), round(position[1] + -1 * scaled_text.height))
  image.paste(im=scaled_text, box=position, mask=scaled_text)

def draw_detections(save_path, image, scored_bboxes_by_class_id, class_id_to_name):
  # Draw all results
  context = ImageDraw.Draw(image, mode="RGBA")
  for class_id, scored_bboxes in scored_bboxes_by_class_id.items():
    # Draw bboxes of one class
    for i in range(scored_bboxes.shape[0]):
      # Draw one bbox
      scored_bbox = scored_bboxes[i, : ]
      class_name = class_id_to_name[class_id]
      text = "%s %1.2f" % (class_name, scored_bbox[4]) # class name + score
      color = list(ImageColor.colormap.values())[class_id + 1]
      draw_box(context=context, corners=scored_bbox[0: 4], color=color)
      draw_text(image=image, text=text, position=(scored_bbox[1], scored_bbox[0]), color=color)

  # Save the image
  image.save(save_path)
  
def draw_gt_detections(save_path, image, scored_bboxes_and_names, class_id_to_name):
  print(scored_bboxes_and_names)
  # Draw all results
  context = ImageDraw.Draw(image, mode="RGBA")
  for i in range(len(scored_bboxes_and_names)):
      scored_bbox_and_name = scored_bboxes_and_names[i]
      scored_bbox = scored_bbox_and_name[0: 5]
      class_id = int(scored_bbox_and_name[5])
      class_name = class_id_to_name[class_id]
      text = "%s %1.2f" % (class_name, scored_bbox[4]) # class name + score
      color = list(ImageColor.colormap.values())[class_id + 1]
      draw_box(context=context, corners=scored_bbox[0: 4], color=color)
      draw_text(image=image, text=text, position=(scored_bbox[1], scored_bbox[0]), color=color)

  # Save the image
  image.save(save_path)