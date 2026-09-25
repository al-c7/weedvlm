# Weed density estimation task

## Task definition

Weed density estimation tasks a VLM with identifying the density of weeds in a
particular image. The VLM will be provided with the possible density categories.
This categorisation is done using a set of weed count, and bounding box area
criteria. The VLM must then select the density category that it thinks applies
to the target weed. The VLM will tested under two image conditions, ones with
bounding boxes around the target weeds, and one without bounding boxes. The
density categorisation rules remain the same between conditions. In this case,
the same source images will be used between each scenario.

## Image formats

There are two image formats for this task, images where all weed species are
annotated, and images without annotations. This means that the VLM must be
able to discern weed species from regular plant/crop species, but not identify
the species in the image.

## Evaluation

The VLM will provide multiple metrics in which it will be evaluated in. The
first one being the density category, alongside the estimated number of
individual weeds, as well as the estimated coverage (in % of the overall image)
that it thinks the weeds cover. The difference of these compared to the ground
truth will be used as the evaluation metrics.

## Provided meta data

Where available from the original dataset, additional meta data will be
provided to the VLM, some of this includes generally regional location (region,
state, country, etc.). Additional data such as time of year, soil conditions,
and the general height at which the image was captured may also be provided
where available.

## Response format

JSON format as specied previously, with `density-category`,
`estimated-number-of-weeds`, and `estimated-weed-coverage` instead of the
`species` field. The `reasoning` field should still be provided.

## Image selection

No stringent requirements on the bounding boxes and species present in this
case. However we will attempt to classify scene density into 'low' and 'high'
density scenes, with some automated method, we will also need to take into
account the overlap between bounding boxes when working out the overall density.
