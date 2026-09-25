# Weed Species ID Task

## Task definition

The purpose of this task is to assess a VLMs ability to identify particular
plant species in an image. For evaluation, there must only ever be a single
unambiguous answer to a particular question. The primary question format for
this task is as follows:

- Multiple choice questions, with 5 total options to select, along with the
  choice to select a 'Negative' option, in which it does not know the answer.

To assess the extent to which the VLM is 'guessing' it's answer, a subset of
the questions will be assessed again in an open-ended format, in which the VLM
is not provided any options to choose from, asides from the option to give up.

## Image formats

There are two image formats used in the weed identification tasks, images
with bounding boxes around the weed of interest, and images that have no
annotations. Images without annotations must only contain a single species
of weed in the image. However, other non-weed species may be present in the
image.

## Evaluation

The Weed species identification task will be evaluated separately for each
image format, each image format will be assessed under both question formats.
The results of each task will be evaluated separately and overall performance
will be compared. The primary response format to evaluate between models is
Accuracy. Additionally, the F1 score will be computed to handle possible class
imbalances within the benchmark.

## Provided meta data

Where available from the original dataset, additional meta data will be
provided to the VLM, some of this includes generally regional location (region,
state, country, etc.). Additional data such as time of year, soil conditions,
and the general height at which the image was captured may also be provided
where available.

## Response format

The VLM will be instructued to provide it's answer in a structured JSON format.
Since many VLMs are focused on writing code, providing their answers in this
format is unlikely to cause any degradations in performance for the VLM itself.
The VLM shall provide 2 fields in it's JSON structured response, a `species`
field that defines the actual species that the VLM has selected. For multiple
choice questions, this should exactly match the option in terms of spelling and
punctuation. An additional `reasoning` field will be required, which features
the post-hoc reasoning for the VLMs decision.

Variations in this will be handled with a manually constructed lookup for
common and scientific names for the plant.

## Image selection

The image requires a number of conditions for each target species to be used as
a species identification question. The main criteria are:

- Minimum total bounding box area and minimum largest bounding box area
- The overall centre of mass of the bounding boxes must be within some limit
  close to the centre of the image, i.e. not in corners.
