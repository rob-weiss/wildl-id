# wildl-id

As a conservationist, hunter, and software developer I've been curious about the insights one can gain from data one has openly access to. This repo contains a collection of `python` tools for wildlife identification and statistical analysis. It is WIP and by no means complete. Please feel free to contact the authors in case of questions.

## Ideas

- Use existing photos of wildlife cameras of ones own hunting grounds to:
  - extract information such as the following using optical character recognition (OCR), e.g. Keras OCR or `pytesseract`:
    - Location
    - Temperature
    - Date and time
  - label the images using the python module `labelImg` and the YOLO format and train a deep neural for automated classification of the species found in the image. In the literatur, not only species have been classified but also individuals identified. Inspired by several papers on sika deer identification Frank Zabel's red deer project, the ultimate goal would be identifying individuals of other species such as the territorial roe deer common all across Germany. As data is limited, transfer learning could be used to tune existing models.
  - visualise statistics of the data, e.g.:
    - Activity periods over the year
    - Correlation of activity and temperature
    - Activity in different locations
