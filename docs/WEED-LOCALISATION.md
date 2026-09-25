# Weed species localisation task

## Task definition

In this task, the VLM is given an image with multiple labelled bounding
boxes, with each numerical label corresponding to a different species of
weed. The VLM is not given the names of any of the weeds, only the bounding
boxes and the labels. The VLM will be asked to identify in which bounding box
X species of weed is located. The VLM must then select a label as its response.

## Image formats

Fine grained weed identification will use a annotated image with a bounding box
around the target weed. In this case, there will be no open-ended questions as
the fundamental task of the question is to differentiate between two visually
similar types of weeds.

Weed species localisation will use images with muliple weed species in them.
Labels will be applyed to the bounding boxes, starting and 1 for the first weed
species, and N for the Nth weed species. Bounding boxes that contain a particular
species of weed will all receive the same label for a given image.

## Evaluation

The primary metric used to evaluate the weed species localisation task
questions will be accuracy. Random guessing should yield and overall accuracy
of reciprocal of the average number of weeds in an image.

To provide a structured baseline to this task, the VLM will be asked a small
set of additional questions that ask it to localise a particular crop species in
an image.

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
The VLM shall provide 2 fields in it's JSON structured response, a `label`
field that defines the actual label that the VLM has selected. For multiple
choice questions, this should exactly match the option in terms of spelling and
punctuation. An additional `reasoning` field will be required, which features
the post-hoc reasoning for the VLMs decision.

## Image selection

The image requires a number of conditions for each target species to be used as
a species identification question. The main criteria are:

- Minimum total bounding box area and minimum largest bounding box area

On top of this, a number of automated and manual review will be performed to
identify visually similar weeds. The automated process will be done by
comparing colours, and well as bounding box areas. These pairs will also be
reviewed manually before being used as questions.
