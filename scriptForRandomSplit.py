import shutil, random, os

#names of images folders
folderList = ['Blueberries', 'Flavored_Water', 'Jelly_Beans_Bag', 
              'Knives', 'Mop_heads_&_Sponges', 'colored_paper_bag', 
              'pasta_sides', 'teacup']
for folder in folderList:
    #create the directories
    os.mkdir('path/to/testFolder' +folder)
    os.mkdir('path/to/trainFolder' +folder)

    dirpath = 'path/to/imageFolder' + folder
    trainDirectory = 'path/to/trainFolder' +folder
    testDirectory = 'path/to/testFolder' +folder
    
    #move the 70% of the images to the train folder
    list = os.listdir(dirpath)
    number = int(0.7 * len(list))
    filenames = random.sample(list, number)
    for fname in filenames:
        srcpath = os.path.join(dirpath, fname)
        shutil.move(srcpath, trainDirectory)
    
    #move the remaining 30% to test folder
    remaining = os.listdir(dirpath)
    for fname in remaining:
        shutil.move(os.path.join(dirpath, fname), testDirectory)
    
        
 
    
'''   
os.mkdir('C:\\Users\\Lorenzo\\Documents\\Didattica Uni\\ArtificialIntelligenceRobotics\\Primo anno\\MachineLearning\\ObjectClassificationHomework\\randtrain\\' +folder)
    os.mkdir('C:\\Users\\Lorenzo\\Documents\\Didattica Uni\\ArtificialIntelligenceRobotics\\Primo anno\\MachineLearning\\ObjectClassificationHomework\\randtest\\' +folder)
    
    dirpath = 'C:\\Users\\Lorenzo\\Documents\\Didattica Uni\\ArtificialIntelligenceRobotics\\Primo anno\\MachineLearning\\ObjectClassificationHomework\\' + folder
    trainDirectory = 'C:\\Users\\Lorenzo\\Documents\\Didattica Uni\\ArtificialIntelligenceRobotics\\Primo anno\\MachineLearning\\ObjectClassificationHomework\\randtrain\\' +folder
    testDirectory = 'C:\\Users\\Lorenzo\\Documents\\Didattica Uni\\ArtificialIntelligenceRobotics\\Primo anno\\MachineLearning\\ObjectClassificationHomework\\randtest\\' +folder

'''