# Weed Localisation

This task asks the VLM to place a bounding box over a specified species of weed that is located in an image. In the question text, the overall template will look like this:

```
There are instances of {target_species} in this image, please give coordinates for bounding boxes of these species in your native format. The response should be as a JSON field and consistent with your model card. The format that you choose to choose to use should be specified very briefly in another JSON field called `response_format`. You should place a bounding box over all instances that you find of {target species}. The field that stores the bounding box coordinates should be called `bboxes`. 
```

The rest of the question should be the same as the other tasks.