# Visual Med-Alpaca Data


Important Notice: These datasets are produced based on [BigBio](https://huggingface.co/bigbio), [ROCO](https://github.com/razorx89/roco-dataset) and [GPT-3.5-Turbo](https://chat.openai.com/chat), and are for academic purpose only.

```
med_alpaca_data_clean.json:
* Size: 44.8MB
* Items: 54412
* Type: json
* Sample:
{
	'instruction': 'i have a tight and painful chest with a dry cough, no fever and no headaches. could it possibly be coronavirus?', 
	'input': '', 
	'output': "Chest tightness and dry cough are some of the symptoms associated with coronavirus (covid-19). it's best to contact your healthcare provider or a medical professional for a proper assessment and testing. they can provide a clinical diagnosis and recommend appropriate treatment options."
},
{
	'instruction': "A mother brings her 3-week-old infant to the pediatrician's office because she is concerned about his feeding habits. He was born without complications and has not had any medical problems up until this time. However, for the past 4 days, he has been fussy, is regurgitating all of his feeds, and his vomit is yellow in color. On physical exam, the child's abdomen is minimally distended but no other abnormalities are appreciated. Which of the following embryologic errors could account for this presentation?", 
	'input': 'Abnormal migration of ventral pancreatic bud, Complete failure of proximal duodenum to recanalize, Error in neural crest cell migration, Abnormal hypertrophy of the pylorus, Failure of lateral body folds to move ventrally and fuse in the midline', 
	'output': 'The correct answer is complete failure of proximal duodenum to recanalize. This embryologic error can result in a condition known as duodenal atresia, where the proximal duodenum is obstructed and prevents the passage of feeds. This can lead to vomiting of bile (yellow in color) and abdominal distension. Treatment typically involves surgical correction of the atresia.'
}
```

```
radiologytraindata_cleaned.csv:
* Size: 12.3MB
* Type: csv
* Items: 65444
* Sample: 
| id         | name                           | caption                                                                                 |
|------------|--------------------------------|-----------------------------------------------------------------------------------------|
| ROCO_00002 | PMC4083729_AMHSR-4-14-g002.jpg | Computed tomography scan in axial view showing obliteration of the left maxillary sinus |
```

```
radiologyvaldata_cleaned.csv:
* Size: 1.5MB
* Type: csv
* Items: 8179
* Sample: 
| id         | name                           | caption                                                                                 |
|------------|--------------------------------|-----------------------------------------------------------------------------------------|
| ROCO_00020 | PMC3970251_CRIONM2014-931546.003.jpg | Axial computed tomography scan of the pelvis showing a diffuse infiltration of the bladder wall, catheter in situ (arrow). |
```

For usage of these datasets, refer to [our code base](https://github.com/cambridgeltl/visual-med-alpaca/tree/main/code).
