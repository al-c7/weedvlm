# Fine grained weed identification task

## Task definition

The purpose of this task is to assess a VLMs ability to distinguish two
visually similar weeds. The VLM will be asked to identify whether a
particular whether two weeds in an image are the same or different
species. In this case, the weed does not asked to identify the species.
The VLM will be able to respond with the following options:

- Same Species
- Different Species
- Negative (Unsure)

## Image formats

Fine grained weed identification will use a annotated image with a bounding box
around the target weeds/plants. In this case, there will be no open-ended
questions as the fundamental task of the question is to differentiate between
two visually similar types of weeds.

## Evaluation

The primary metric used to evaluate the fine-grained weed identification
questions will be accuracy. Random guessing should yield and overall accuracy
of about 0.5.

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
The VLM shall provide 2 fields in it's JSON structured response, an `option`
field that defines which of the options the VLM has selected. For multiple
choice questions, this should exactly match the option in terms of spelling and
punctuation. An additional `reasoning` field will be required, which features
the post-hoc reasoning for the VLMs decision.

## Image selection

The image requires a number of conditions for each target species to be used as
a species identification question. The main criteria are:

- Minimum total bounding box area and minimum largest bounding box area
- The overall centre of mass of the bounding boxes must be within some limit
  close to the centre of the image, i.e. not in corners.

On top of this, a number of automated and manual review will be performed to
identify visually similar weeds. The automated process will be done by
comparing colours, and well as bounding box areas. These pairs will also be
reviewed manually before being used as questions. But for now we will just be
using bounding box sizes.
