from pathlib import Path
from skimage.segmentation import random_walker
import numpy as np

patient_dest_dir = Path("/Users/Bram/Documents/RP/miccai_data_numpy/part1/0522c0002")

img = np.load(patient_dest_dir.joinpath("img.npy"))
structures = np.load(patient_dest_dir.joinpath("structures.npy")).astype(int)
structures[structures != 3] = 0
print(len(img),len(img[0]),len(img[0,0]))



# one voxel as seed point for each label
# markers = [[],[74, 239, 260],[85, 217, 259],[65, 185, 232],[86, 197, 277],[86, 187, 241],[65, 217, 302],[64, 218, 219],[58, 201, 287],[57, 205, 236],[61, 209, 229]]
test = np.zeros(structures.shape).astype(int)


mandible = []
background = []
seeds = [[162.24053657068237, 212.5008276493399], [287.64306426848304, 279.2981539607312], [154.0484116457004, 334.43745634041744], [210.44804093692233, 274.5719280424724], [177.99462296487843, 220.37787084643793], [179.57003160429804, 228.56999577141988], [161.92545484279844, 282.76405296745435], [182.72084888313725, 283.7092981511061], [177.36445950911042, 197.37690471091167], [210.44804093692215, 201.78804890128657], [240.69588681377857, 237.07720242428576], [246.68243964357356, 277.0925818655437], [208.872632297503, 296.3125672664629], [213.59885821576128, 262.28374065499946], [206.66706020231527, 160.5123425484929], [94.4979650756393, 181.62281831671564], [92.29239298045185, 248.10506290022303], [101.74484481696949, 311.4364902048912], [142.07530598611103, 352.082033101917], [195.954281454262, 362.16464839420246], [252.03882901759994, 328.45090351062294], [269.0532423233317, 285.91487024629356], [273.7794682415905, 245.58440907715163], [249.20309346664465, 238.33752933582144], [169.80249803989648, 242.74867352619634], [181.46052197160157, 220.37787084643793], [216.43459376671683, 217.85721702336656], [231.24343497726113, 256.2971878252049], [184.61133925044078, 266.6948848453743], [177.04937778122667, 296.6276489943469], [167.91200767259295, 318.9984516741053], [138.60940697938827, 291.90142307608807], [120.01958503423693, 241.17326488677674], [272.5191413300548, 285.2847067905257], [243.21654063685014, 226.0493419483485], [247.62768482722532, 285.5997885184096], [212.3385313042261, 263.859149294419], [183.0359306110214, 247.47489944445513], [154.36349337358462, 269.8457021242135], [125.69105613614724, 243.37883698196418], [172.32315186296785, 223.84376985316106], [185.87166616197646, 223.21360639739322], [189.65264689658352, 187.92445287439404], [257.0801366637427, 205.56902963589363], [225.57196387535055, 245.2693273492677], [172.9533153187357, 271.1060290357492], [180.83035851583372, 290.0109327087845], [177.3644595091106, 311.7515719327751], [154.99365682935218, 309.23091810970374], [214.85918512729722, 306.39518255874844], [217.69492067825252, 287.17519715782925], [216.74967549460075, 201.47296717340265], [216.74967549460075, 216.59689011183087], [195.95428145426195, 197.06182298302775], [188.39231998504783, 198.95231335033128], [188.39231998504783, 215.96672665606303], [190.91297380811898, 240.54310143100886], [189.65264689658352, 266.69488484537436], [184.29625752255686, 284.9696250626418], [182.09068542736946, 301.98403836837355], [161.92545484279802, 315.8476343952661], [182.4057671552534, 258.8178416482763], [156.56906546877144, 255.66702436943712], [168.85725285624474, 221.00803430220577], [154.04841164570044, 213.44607283299166], [240.69588681377883, 277.7227453213116], [218.32508413402041, 247.78998117233908], [204.14640637924356, 269.53062039632965], [222.73622832439477, 290.9561778924363], [206.98214193019885, 202.41821235705441], [217.0647572224843, 225.10409676469675], [198.1598535494494, 232.03589477814302], [186.5018296177443, 209.6650920983846], [191.85821899177097, 196.4316595272599], [165.39135384952158, 200.52772198975092], [150.26743091109336, 217.22705356759874], [180.20019506006588, 231.7208130502591], [182.7208488831373, 285.2847067905258], [179.88511333218202, 259.1329233761603], [142.39038771399535, 274.88700977035637], [162.87070002645024, 281.1886443280348], [153.73332991781652, 308.60075465393595], [173.2683970466202, 313.9571440279626], [187.44707480139667, 321.2040237692928], [200.68050737252136, 317.1079613068018], [206.03689674654785, 292.5315865318559], [195.95428145426237, 288.43552406936493], [196.58444491003016, 266.0647213896065], [226.8322907868866, 266.37980311749044], [202.25591601194043, 198.00706816667952], [212.65361303211006, 210.61033728203637], [207.61230538596743, 229.83032268295557], [193.74870935907484, 227.62475058776812], [186.1867478898607, 211.87066419357205], [181.14544024371764, 201.15788544551873], [169.48741631201256, 202.73329408493834], [154.99365682935257, 213.44607283299166], [162.2405365706828, 222.8985246695093], [175.4739691418075, 226.0493419483485], [164.44610866587036, 284.9696250626418], [177.679541236994, 291.9014230760881], [198.79001700521673, 290.3260144366685], [210.448040936923, 298.51813936165047], [198.7900170052173, 319.3135334019893], [185.87166616197652, 318.0532064904536], [165.3913538495216, 310.8063267491234], [150.26743091109338, 292.8466682597399], [209.18771402538664, 205.88411136377758], [197.21460836579763, 235.18671205698226], [178.9398681485302, 233.92638514544657], [153.41824818993257, 207.45952000319718], [183.03593061102063, 199.89755853398307], [176.41921432545806, 288.12044234148107], [200.9955891004039, 298.20305763376655], [200.68050737252173, 315.53255266738216], [171.06282495143316, 319.94369685775706], [161.9254548427994, 314.2722257558465], [151.842839550513, 294.4220768991595], [194.06379108695847, 199.89755853398307], [204.7765698350118, 210.92541900992032], [198.7900170052173, 228.88507749930383], [183.03593061102058, 227.62475058776815], [159.7198827476104, 225.10409676469678], [159.71988274761102, 211.24050073780424], [182.09068542736946, 205.25394790800974], [172.00807013508395, 289.38076925301675], [184.2962575225569, 292.8466682597399], [205.09165156289572, 298.8332210895344], [200.05034391675207, 318.3682882183375], [189.65264689658267, 316.16271612315006], [172.00807013508475, 314.9023892116144], [154.0484116457004, 303.2443652799093], [151.84283955051296, 288.7506057972489], [180.83035851583372, 204.30870272435797], [202.25591601194037, 202.10313062917052], [208.5575505696188, 214.70639974452737], [183.3510123389051, 230.14540441083952], [156.56906546877178, 210.6103372820364], [156.8841471966557, 228.5699957714199], [159.40480101972707, 278.3529087770795], [176.7342960533428, 284.65454333475793], [205.09165156289572, 290.64109616455244], [206.35197847443212, 305.1348556472128], [186.1867478898612, 323.40959586448025], [163.81594521010197, 312.0666536606591], [149.32218572744222, 296.6276489943469], [173.58347877450396, 199.89755853398304], [202.8860794677086, 200.5277219897509], [206.35197847443177, 216.9119718397148], [169.48741631201204, 243.06375525408026], [145.54120499283408, 217.22705356759872], [156.25398374088726, 281.50372605591866], [177.36445950911, 286.2299519741775], [200.68050737252076, 292.53158653185596], [206.03689674654743, 317.1079613068018], [189.96772862446744, 318.0532064904536], [154.99365682935218, 308.6007546539359], [170.74774322354824, 204.30870272435794], [201.3106708282886, 200.8428037176348], [204.46148810712782, 226.99458713200028], [156.56906546877178, 236.76212069640184], [159.71988274761102, 275.20209149824024], [191.85821899177103, 284.33946160687395], [206.66706020231533, 311.12140847700726], [160.35004620337887, 318.6833699462214], [173.89856050238748, 207.77460173108108], [203.83132465135998, 201.78804890128657], [201.62575255617202, 235.8168755127501], [154.67857510146774, 236.76212069640187], [155.30873855723547, 271.73619249151716], [178.9398681485302, 282.4489712395705], [209.81787748115454, 301.03879318472184], [189.65264689658449, 330.34139387792646], [162.87070002645115, 313.95714402796256], [179.2549498764147, 204.30870272435794], [201.94083428405708, 202.73329408493834], [205.09165156289637, 224.78901503681283], [148.06185881590739, 227.30966893500556], [152.47300300628262, 279.29815403585263], [188.07723825716212, 271.4211108387546], [206.66706020231533, 295.9974856137004], [193.43362763119063, 318.68337002134274], [158.14447410819344, 311.7515720078965], [173.26839704662095, 204.3087027994793], [194.0637910869574, 202.1031306291705], [206.66706020231675, 228.88507757442517], [179.5700316042981, 237.0772024994071], [153.10316646204868, 234.87163040421964], [147.11661363225417, 274.57192811759376], [204.14640637924387, 292.5315866820986], [177.3644595091106, 282.764053117697], [168.8572528562447, 314.58730763397307], [170.74774322354824, 208.08968360920767], [196.26936318214587, 201.78804905152924], [196.26936318214587, 233.29622183992137], [169.4874163120141, 242.11851022067117], [150.5825126389788, 238.96769294183196], [146.8015319043719, 263.5440677167778], [172.00807013508393, 275.2020916484829], [188.0772382571639, 288.4355242196076], [199.42018046098508, 316.1627162733927], [177.04937778122869, 316.1627162733927], [156.56906546877178, 304.18961061380367], [137.6641617957365, 293.16175010030577], [158.1444741081914, 211.24050085048623], [184.9264209783247, 200.5277221024329], [195.95428145426195, 203.3634576533882], [192.1733007196549, 228.88507761198582], [168.22708940047687, 243.06375536676228], [140.18481561880787, 233.92638525812856], [130.41728205440631, 273.62668297150265], [154.99365682935218, 269.21553878112775], [184.9264209783247, 289.69585109358263], [201.62575255617253, 313.0118989569928], [169.80249803989648, 316.47779796371594], [145.54120499283454, 302.9292836647073], [164.76119039375374, 206.1991932043435], [181.7756036994855, 203.67853938127212], [201.3106708282886, 210.61033739471839], [179.57003160429804, 241.17326499945875], [155.3087385572361, 250.31063510809247], [137.34908006785258, 255.66702448211913], [157.51431065242357, 266.37980323017246], [172.00807013508393, 282.764053117697], [201.94083428405645, 298.5181395118931], [204.1464063792439, 308.28567307629464], [167.59692594470903, 313.9571441782052], [136.40383488420082, 297.2578126003574], [112.77270529290674, 276.7775002879025], [102.37500827273733, 255.03686106391194], [120.01958503423693, 229.83032283319824], [139.2395704351561, 215.9667268063057], [170.1175797677804, 205.88411151402022], [128.21170995921997, 222.58344309186802], [117.18384948328301, 280.2433992946256], [130.10220032652367, 255.98210624756365], [167.91200767259167, 261.96865907735815], [172.63823359085077, 205.8841115140202], [201.31067082828758, 212.18574607169862], [136.7189166120835, 254.09161588026012], [172.32315186296645, 283.7092983013487], [196.26936318214436, 312.06665381090164], [165.70643557740397, 311.4364903551338], [113.4028687486732, 279.2981541109738], [104.26549864003961, 253.46145242449228], [121.91007540153933, 226.679505554359], [169.17233458412764, 205.25394805825235], [169.1723345841275, 231.40573147261782], [139.23957043515486, 255.3519427917958], [157.5143106524222, 270.79094745810795], [192.48838244753736, 297.57289432824126], [192.48838244753728, 313.0118989945534], [165.70643557740397, 311.7515720830177], [131.99269069382447, 295.99748568882166], [113.71795047655712, 278.9830723830899], [103.00517172850516, 252.20112551295665], [128.21170995921887, 223.2136065476359], [180.20019506006585, 201.78804905152924], [193.74870935907447, 210.61033743227904], [192.17330071965486, 225.73426037070726], [153.41824818993257, 242.4335919485551], [134.5133445168973, 261.6535773494743], [156.25398374088786, 284.0243800292327], [179.2549498764141, 297.2578126003574], [191.54313726388702, 309.5459999878303], [172.63823359085177, 313.6420624503213], [153.7333299178165, 308.60075480417856], [109.62188801406752, 275.2020916484829], [107.731397646764, 240.22801985336764], [136.40383488420082, 217.5421354457253], [170.43266149566432, 205.25394805825238], [192.17330071965486, 206.51427496978806], [193.43362763119055, 219.1175440851449], [165.39135384952158, 240.22801985336764], [135.45858970054906, 256.61226970333155], [174.52872395815646, 285.59978866865225], [154.04841164570166, 304.50469234168753], [102.3750082727384, 263.22898598889384], [162.24053657068328, 218.48738062937704], [189.02248344081653, 208.08968360920764], [179.570031604299, 236.13195739087664], [148.37694054379082, 244.9542457716264], [134.19826278901445, 261.96865907735815], [152.15792127839805, 286.2299521244201], [151.52775782263026, 298.2030577840091], [104.5805803679259, 276.46241856001853], [99.53927272178308, 256.9273514312154], [116.23860429963078, 225.1040969149394], [153.41824818993348, 220.06278926879665], [122.5402388573083, 225.41917864282337], [141.44514253034356, 241.17326503701946], [133.2530176053616, 257.8725966148673], [144.28087808129885, 296.6276491445896], [109.93696974195144, 281.8188079340453], [104.26549864004086, 269.21553881868846], [144.59595980918277, 225.7342603707073], [134.19826278901425, 253.4614524244924], [125.06089268038059, 265.7496398119653], [139.55465216304103, 285.28470694076844], [104.58058036792568, 263.8591494446618], [106.15598900734439, 250.94079860142102], [120.96483021788869, 242.74867367643907], [111.51237838137105, 266.6948849956171], [120.96483021788869, 249.0503082341175], [113.7179504765585, 261.96865907735827], [114.03303220444346, 253.7765341523762], [106.78615246311466, 242.748673676439], [129.7871185986373, 239.59785639759977], [152.47300300628228, 230.14540456108216], [146.17136844860406, 259.44800525428684], [126.63630131979927, 285.9148703965362], [255.18964629643915, 240.22801985336758], [183.98117579467294, 218.80246235726096]]
seeds = np.array(seeds).astype(int)
for i in range(59,90):
    for point in seeds:
        if structures[i][point[0]][point[1]] == 3:
            mandible.append([i,point[0],point[1]])
        else:
            background.append([i,point[0],point[1]])

for point in mandible:
    test[point[0],point[1],point[2]] = 1
for point in background:
    test[point[0],point[1],point[2]] = 2

img = img[59:90,100:300,180:340]
structures = structures[59:90,100:300,180:340]
test = test[59:90,100:300,180:340]
labels = random_walker(img, test, beta=10, mode='cg_j', spacing = [3.938,1.000,1.000])

use_napari = True

if not use_napari:
    import matplotlib.pyplot as plt

    slice_num = 100
    fig, ax = plt.subplots(layout="tight", figsize=(5, 5))
    ax.imshow(img[slice_num], cmap="gray")
    ax.imshow(structures[slice_num], alpha=(structures[slice_num] > 0).astype(float))
    ax.set_axis_off()
    plt.show()

else:
    import napari

    viewer = napari.Viewer()
    viewer.add_image(img, name="CT scan", colormap="gray", interpolation2d="bicubic")

    viewer.add_labels(structures, name="segmentation")
    viewer.add_labels(labels, name="result of rw")
    napari.run()

# takes an 3D image, returns a graph for random

#def process_input():
#    return

#def random_walker():
#    return

# takes a 3D image, returns the segmentation (labels, uncertainties, etc.)
def segment():
    return

