import numpy as np
import matplotlib.pyplot as plt
import RegLib2022 as Reglib

ImFix = Reglib.OpImage("Lungs_Fixed.png")
ImDef = Reglib.OpImage("Lungs_Fixed.png")
ImMov = Reglib.OpImage("Lungs_Moving.png")

# Montrer l'image en mouvement au debut
ImMov.show()

[SizeX,SizeY] = ImFix.size()

#transformation affine
[DefX,DefY] = Reglib.GenerateNullDisplacementField("Lungs_Fixed.png") #initialisation des matrices de deformations
for i in range(SizeX):
    for j in range(SizeY):
        DefX.put(10*np.cos(i/30),i,j)
        DefY.put(20*np.cos(j/30),i,j)
        
Reglib.TransportImage(DefX.data,DefY.data,ImMov,ImDef)
ImDef.show()
ImDef.SaveImage(filename="ImDef_transformee.png")
ImFix.SaveComparisonWithAnotherImage(ImDef, LabelSelfIm="Fixed Image", LabelComparedIm="Compared Image", filename = "ComparedImage.png")

### TRANSFORMATION LOCALE
ImFix=Reglib.OpImage('Lungs_Fixed.png')
ImDef=Reglib.OpImage('Lungs_Fixed.png') 
ImMov=Reglib.OpImage('Lungs_Moving.png')

def CptGradE(ImFix,ImDef,UpDefX,UpDefY):
    """
    -> ImFix and ImDef are inputs and won't be modified
    -> Energy gradients (equivalent to forces here) will be stored in UpDefX,UpDefY
    """

    ImDefGrad=ImDef.grad()

    tmpMat=ImFix.data[:,:]-ImDef.data[:,:]
    UpDefX.data[:,:]=np.multiply(ImDefGrad[0],tmpMat)
    UpDefY.data[:,:]=np.multiply(ImDefGrad[1],tmpMat)

    UpDefX.data[:,:]=-2*UpDefX.data[:,:]
    UpDefY.data[:,:]=-2*UpDefY.data[:,:]
    

#ce champ de vecteurs 2D est stoque dans deux images de meme taille que ImFix mais aurait tres bien pu etre stoque dans
#un np.array de taille  (ImFix.shape[0],ImFix.shape[1],2).
[DefX,DefY]=Reglib.GenerateNullDisplacementField('Lungs_Fixed.png')
[UpdateDefX,UpdateDefY]=Reglib.GenerateNullDisplacementField('Lungs_Fixed.png')



Reglib.TransportImage(DefX.data,DefY.data,ImMov,ImDef)
E_init=Reglib.Cpt_SSD(ImDef,ImFix)
print('Initial energy:'+str(E_init))

DefX.show(LabelImage='DefX')
DefY.show(LabelImage='DefY')
ImFix.CompareWithAnotherImage(ImMov,LabelSelfIm='Fixed image',LabelComparedIm='Moving image',ShowAll=1)


#Minimisation de l'energie E
lr=1000.
conv_Energy=[E_init]

for iter in range(150):
    CptGradE(ImFix,ImDef,UpdateDefX,UpdateDefY)
    UpdateDefX.GaussianFiltering(20.)
    UpdateDefY.GaussianFiltering(20.)

    np_DefX=DefX.data
    np_DefY=DefY.data
    np_UpdateDefX=UpdateDefX.data
    np_UpdateDefY=UpdateDefY.data

    np_DefX=np_DefX-lr*np_UpdateDefX
    np_DefY=np_DefY-lr*np_UpdateDefY

    DefX.putToAllPoints(np_DefX)
    DefY.putToAllPoints(np_DefY)

    Reglib.TransportImage(DefX.data,DefY.data,ImMov,ImDef)
    #ImFix.CompareWithAnotherImage(ImDef,LabelSelfIm='Fixed image',LabelComparedIm='Deformed image')
    E_current=Reglib.Cpt_SSD(ImDef,ImFix)
    conv_Energy.append(E_current)

    print('iteration:',iter,'  ->  energy=',E_current)

#convergence
plt.plot(conv_Energy)
plt.show()