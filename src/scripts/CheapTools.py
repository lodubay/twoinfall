"""
Chemical Evolution Analytic Package (CHEAP)
Created by P.A. Palicio for publication
Palicio et al. (2023), A&A 678, A61, doi:10.1051/0004-6361/202346567
Code source: https://bitbucket.org/pedroap/cheap/src/master/
"""

import numpy as np
from scipy.special import erf, expi

# -----------------------------------------------
#info
# -----------------------------------------------
#heaviside
#_safelog10
#_safelog
#_safeexpi
#GaussWeightsAndNodes
#FromSigmaToAbundance
# -----------------------------------------------
#Get_psi
#Get_ipsi
#Get_DTD
#Get_DTD_arr
#Get_infall
# -----------------------------------------------
#R1a_numeric
#R1a_analytic_gaussian
#R1a_analytic_exponential
#R1a_analytic_inverse
#R1a_analytic
#iR1a_analytic
#Get_CIa
# -----------------------------------------------
#SolveChemEvolModel_GaussianTerm
#SolveChemEvolModel_ExponentialTerm
#SolveChemEvolModel_InverseTerm
#SolveChemEvolModel_InhomogeneousTrivialTerm
#SolveChemEvolModel
# -----------------------------------------------
#ChemicalSolutionVerifier
# -----------------------------------------------
#add_element
#prepare_chemdict
#QD
#PD
#SD
#QD_gorro
#SD_gorro
# -----------------------------------------------
#LatexCheckEqA9a
#LatexCheckEqA9b
#LatexCheckEqA9c
# -----------------------------------------------
#Load_MR01_dict
#Load_G05Wide_dict
#Load_G05Close_dict
#Load_P08_dict
#Load_T08_dict
#Load_S05_dict
#Load_MVP06_dict
# -----------------------------------------------

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# 				 INFO FUNCTIONS
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# info -----------------------------------------
def info():
	'''Prints info about the version'''
	txt = "Version 1.24.05.21: Added Mg yields from Johnson & Weinberg (2020)"
	return(txt)
# -----------------------------------------



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# 				 AUX FUNCTIONS
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# heaviside ------------------------
def heaviside(x): return(1.*(x>0));
# ----------------------------------

# _safelog10--------------------------
def _safelog10(num, den):
	not_null_den = den!=0
	output = np.zeros_like(num)
	output[not_null_den] = np.log10(num[not_null_den]/den[not_null_den])
	
	zero_num = (num==0)
	output[zero_num] = 0.

	null_den = (den==0)
	output[null_den] = np.nan
	return(output)
# ---------------------------------


# _safelog--------------------------
def _safelog(num, den):
	not_null_den = den!=0
	output = np.zeros_like(num)
	output[not_null_den] = np.log(num[not_null_den]/den[not_null_den])
	
	zero_num = (num==0)
	output[zero_num] = 0.

	null_den = (den==0)
	output[null_den] = np.nan
	return(output)
# ---------------------------------

# _safeexpi--------------------------------------------
def _safeexpi(x): return( (x>0)*expi(x + 1.*(x<=0)) );# Avoid the divergence when expi(0)
# -----------------------------------------------------


# GaussWeightsAndNodes---------------------------------------------------------------------
def GaussWeightsAndNodes(order=64):
	''' Return the weights and nodes for the Gaussian Quadrature method
	weights, nodes = GaussWeightsAndNodes(order=100)'''
	if order==20:
		Gauss_nodes = [-0.148874338981631, 0.148874338981631, -0.433395394129247, 0.433395394129247, -0.679409568299024, 0.679409568299024, -0.865063366688985, 0.865063366688985, -0.973906528517172, 0.973906528517172]
		weights = [0.295524224714753, 0.295524224714753, 0.269266719309996, 0.269266719309996, 0.219086362515982, 0.219086362515982, 0.149451349150581, 0.149451349150581, 0.066671344308688, 0.066671344308688]

	elif order==40:
		weights = [0.0775059479784248,0.0775059479784248,0.077039818164248,0.077039818164248,0.0761103619006262,0.0761103619006262,0.0747231690579683,0.0747231690579683,0.0728865823958041,0.0728865823958041,0.0706116473912868,0.0706116473912868,0.0679120458152339,0.0679120458152339,0.064804013456601,0.064804013456601,0.0613062424929289,0.0613062424929289,0.0574397690993916,0.0574397690993916,0.0532278469839368,0.0532278469839368,0.0486958076350722,0.0486958076350722,0.0438709081856733,0.0438709081856733,0.038782167974472,0.038782167974472,0.0334601952825478,0.0334601952825478,0.0279370069800234,0.0279370069800234,0.022245849194167,0.022245849194167,0.0164210583819079,0.0164210583819079,0.0104982845311528,0.0104982845311528,0.0045212770985332,0.0045212770985332]
		Gauss_nodes = [-0.0387724175060508,0.0387724175060508,-0.1160840706752552,0.1160840706752552,-0.1926975807013711,0.1926975807013711,-0.2681521850072537,0.2681521850072537,-0.3419940908257585,0.3419940908257585,-0.413779204371605,0.413779204371605,-0.4830758016861787,0.4830758016861787,-0.5494671250951282,0.5494671250951282,-0.6125538896679802,0.6125538896679802,-0.6719566846141796,0.6719566846141796,-0.7273182551899271,0.7273182551899271,-0.7783056514265194,0.7783056514265194,-0.8246122308333117,0.8246122308333117,-0.8659595032122595,0.8659595032122595,-0.9020988069688743,0.9020988069688743,-0.9328128082786765,0.9328128082786765,-0.9579168192137917,0.9579168192137917,-0.9772599499837743,0.9772599499837743,-0.990726238699457,0.990726238699457,-0.9982377097105593,0.9982377097105593]		
		
	elif order==100:
		Gauss_nodes = [-0.0243502926634244, 0.0243502926634244, -0.072993121787799, 0.072993121787799, -0.1214628192961206, 0.1214628192961206, -0.1696444204239928, 0.1696444204239928, -0.2174236437400071, 0.2174236437400071, -0.2646871622087674, 0.2646871622087674, -0.311322871990211, 0.311322871990211, -0.3572201583376681, 0.3572201583376681, -0.4022701579639916, 0.4022701579639916, -0.4463660172534641, 0.4463660172534641, -0.489403145707053, 0.489403145707053, -0.5312794640198946, 0.5312794640198946, -0.571895646202634, 0.571895646202634, -0.6111553551723933, 0.6111553551723933, -0.6489654712546573, 0.6489654712546573, -0.6852363130542333, 0.6852363130542333, -0.7198818501716109, 0.7198818501716109, -0.7528199072605319, 0.7528199072605319, -0.7839723589433414, 0.7839723589433414, -0.8132653151227975, 0.8132653151227975, -0.8406292962525803, 0.8406292962525803, -0.8659993981540928, 0.8659993981540928, -0.8893154459951141, 0.8893154459951141, -0.9105221370785028, 0.9105221370785028, -0.9295691721319396, 0.9295691721319396, -0.9464113748584028, 0.9464113748584028, -0.9610087996520538, 0.9610087996520538, -0.973326827789911, 0.973326827789911, -0.983336253884626, 0.983336253884626, -0.9910133714767443, 0.9910133714767443, -0.9963401167719553, 0.9963401167719553, -0.9993050417357722, 0.9993050417357722]

		weights = [0.0486909570091397, 0.0486909570091397, 0.0485754674415034, 0.0485754674415034, 0.048344762234803, 0.048344762234803, 0.0479993885964583, 0.0479993885964583, 0.0475401657148303, 0.0475401657148303, 0.04696818281621, 0.04696818281621, 0.0462847965813144, 0.0462847965813144, 0.0454916279274181, 0.0454916279274181, 0.0445905581637566, 0.0445905581637566, 0.0435837245293235, 0.0435837245293235, 0.0424735151236536, 0.0424735151236536, 0.0412625632426235, 0.0412625632426235, 0.0399537411327203, 0.0399537411327203, 0.0385501531786156, 0.0385501531786156, 0.03705512854024, 0.03705512854024, 0.0354722132568824, 0.0354722132568824, 0.0338051618371416, 0.0338051618371416, 0.0320579283548516, 0.0320579283548516, 0.0302346570724025, 0.0302346570724025, 0.0283396726142595, 0.0283396726142595, 0.0263774697150547, 0.0263774697150547, 0.0243527025687109, 0.0243527025687109, 0.0222701738083833, 0.0222701738083833, 0.0201348231535302, 0.0201348231535302, 0.0179517157756973, 0.0179517157756973, 0.0157260304760247, 0.0157260304760247, 0.0134630478967186, 0.0134630478967186, 0.0111681394601311, 0.0111681394601311, 0.0088467598263639, 0.0088467598263639, 0.0065044579689784, 0.0065044579689784, 0.0041470332605625, 0.0041470332605625, 0.0017832807216964, 0.0017832807216964]
	
	else:

		Gauss_nodes = [-0.99971372677344123367822847, -0.99849195063959581840016336, -0.99629513473312514918613173, -0.99312493703744345965200989, -0.98898439524299174800441875, -0.98387754070605701549610016, -0.97780935848691828855378109, -0.97078577576370633193089786, -0.96281365425581552729365933, -0.95390078292549174284933693, -0.94405587013625597796277471, -0.93328853504307954592433367, -0.92160929814533395266695133, -0.90902957098252969046712634, -0.89556164497072698669852102, -0.88121867938501841557331683, -0.86601468849716462341074, -0.84996452787959128429336259, -0.83308387988840082354291583, -0.8153892383391762543939888, -0.7968978923903144763895729, -0.7776279096494954756275514, -0.757598118519707176035668, -0.73682808980202070551242772, -0.71533811757305644645996712, -0.69314919935580196594864794, -0.67028301560314101580258701, -0.6467619085141292798326303, -0.62260886020370777160419085, -0.59784747024717872126480655, -0.57250193262138119131687044, -0.54659701206509416746799426, -0.52015801988176305664681575, -0.4932107892081909335693088, -0.4657816497733580422492166, -0.43789740217203151310897804, -0.4095852916783015425288684, -0.3808729816246299567633625, -0.35178852637242172097234383, -0.32236034390052915172247658, -0.2926171880384719647375559, -0.26258812037150347916892934, -0.23230248184497396964950996, -0.20178986409573599723604886, -0.17108008053860327488753238, -0.14020313723611397320751461, -0.10918920358006111500342601, -0.07806858281343663669481737, -0.046871682421591631614923913, -0.0156289844215430828722167, 0.0156289844215430828722167, 0.04687168242159163161492391, 0.07806858281343663669481737, 0.10918920358006111500342601, 0.14020313723611397320751461, 0.17108008053860327488753238, 0.20178986409573599723604886, 0.23230248184497396964950996, 0.2625881203715034791689293, 0.29261718803847196473755589, 0.3223603439005291517224766, 0.35178852637242172097234383, 0.38087298162462995676336255, 0.4095852916783015425288684, 0.437897402172031513108978, 0.4657816497733580422492166, 0.4932107892081909335693088, 0.52015801988176305664681575, 0.54659701206509416746799426, 0.57250193262138119131687044, 0.59784747024717872126480655, 0.62260886020370777160419085, 0.6467619085141292798326303, 0.67028301560314101580258701, 0.69314919935580196594864794, 0.71533811757305644645996712, 0.73682808980202070551242772, 0.75759811851970717603566796, 0.77762790964949547562755139, 0.79689789239031447638957288, 0.81538923833917625439398876, 0.83308387988840082354291583, 0.84996452787959128429336259, 0.86601468849716462341073997, 0.88121867938501841557331683, 0.89556164497072698669852102, 0.90902957098252969046712634, 0.92160929814533395266695133, 0.93328853504307954592433367, 0.94405587013625597796277471, 0.95390078292549174284933693, 0.96281365425581552729365933, 0.97078577576370633193089786, 0.97780935848691828855378109, 0.98387754070605701549610016, 0.98898439524299174800441875, 0.99312493703744345965200989, 0.99629513473312514918613173, 0.99849195063959581840016336, 0.99971372677344123367822847]
		weights = [7.3463449050567173040632E-4, 0.0017093926535181052395294, 0.0026839253715534824194396, 0.00365596120132637518234246, 0.004624450063422119351095789, 0.00558842800386551515721195, 0.0065469484508453227641521, 0.00749907325546471157882874, 0.00844387146966897140262083, 0.00938041965369445795141824, 0.0103078025748689695857821, 0.01122511402318597711722157, 0.0121314576629794974077448, 0.01302594789297154228555858, 0.01390771070371877268795415, 0.01477588452744130176887999, 0.01562962107754600272393687, 0.01646808617614521264310498, 0.0172904605683235824393442, 0.0180959407221281166643908, 0.01888373961337490455294117, 0.01965308749443530586538147, 0.02040323264620943276683885, 0.0211334421125276415426723, 0.02184300241624738631395374, 0.022531220256336272701796971, 0.02319742318525412162248885, 0.0238409602659682059625604, 0.024461202707957052719975, 0.0250575444815795897037642, 0.02562940291020811607564201, 0.02617621923954567634230874, 0.02669745918357096266038466, 0.02719261344657688013649157, 0.02766119822079238829420416, 0.02810275565910117331764833, 0.02851685432239509799093676, 0.0289030896011252031348762, 0.02926108411063827662011902, 0.02959048805991264251175451, 0.02989097959333283091683681, 0.0301622651051691449190687, 0.03040407952645482001650786, 0.03061618658398044849645944, 0.0307983790311525904277139, 0.0309504788504909882340635, 0.03107233742756651658781017, 0.03116383569620990678381832, 0.0312248842548493577323765, 0.0312554234538633569476425, 0.03125542345386335694764247, 0.0312248842548493577323765, 0.0311638356962099067838183, 0.03107233742756651658781017, 0.0309504788504909882340635, 0.0307983790311525904277139, 0.03061618658398044849645944, 0.03040407952645482001650786, 0.03016226510516914491906868, 0.02989097959333283091683681, 0.0295904880599126425117545, 0.029261084110638276620119, 0.02890308960112520313487623, 0.0285168543223950979909368, 0.0281027556591011733176483, 0.0276611982207923882942042, 0.0271926134465768801364916, 0.02669745918357096266038466, 0.02617621923954567634230874, 0.025629402910208116075642, 0.0250575444815795897037642, 0.024461202707957052719975, 0.02384096026596820596256041, 0.0231974231852541216224889, 0.022531220256336272701797, 0.0218430024162473863139537, 0.0211334421125276415426723, 0.0204032326462094327668389, 0.0196530874944353058653815, 0.0188837396133749045529412, 0.01809594072212811666439075, 0.0172904605683235824393442, 0.01646808617614521264310498, 0.01562962107754600272393687, 0.01477588452744130176888, 0.013907710703718772687954149, 0.0130259478929715422855586, 0.0121314576629794974077448, 0.01122511402318597711722157, 0.0103078025748689695857821, 0.00938041965369445795141824, 0.0084438714696689714026208, 0.00749907325546471157882874, 0.0065469484508453227641521, 0.00558842800386551515721195, 0.00462445006342211935109579, 0.003655961201326375182342459, 0.00268392537155348241943959, 0.001709392653518105239529358, 7.34634490505671730406321E-4]

	weights = np.array(weights)
	Gauss_nodes = np.array(Gauss_nodes)

	return(weights, Gauss_nodes)
# ---------------------------------------------------------------------------------

# FromSigmaToAbundance---------------------------------------------------------------------
def FromSigmaToAbundance(t, sigmaX, chemdict):
	'''Computes the abundance ratio [X/H] at the time t from the density of the X-element sigmaX.''' 
	sigma_gas = Get_psi(t, chemdict)/chemdict["nuL"];
	abundance = _safelog10(sigmaX, sigma_gas)
	return(abundance)	
# -------------------------# End of FromSigmaToAbundance ----------------------------------


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# 				 MODEL FUNCTION DEFINITIONS
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# psi ---------------------------------------------------------------------------------
def Get_psi(t, chemdict):
	'''Psi function Get_psi(t, chemdict)
	Returns the SFR'''
	# Extract the values of the parameters:
	omega = chemdict["omega"];
	R = chemdict["R"];
	nuL = chemdict["nuL"];
	tauj = chemdict["tauj"];
	tj = chemdict["tj"];
	assert(len(tauj)==len(tj)), "tj, tauj lengths mismatch"
	Aj = chemdict["Aj"]
	assert(len(tauj)==len(Aj)), "Aj, tauj lengths mismatch"
	sigma_gas_0 = chemdict["sigma_gas_0"]

	# Useful definitions
	alpha = (1.+omega-R)*nuL
	inv_alpha = 1./alpha;
	N = len(tj)

	# Evaluate each infall
	value = 0.;
	for j in range(N):
		deltatj = t-tj[j];
		if tauj[j]!=inv_alpha:
			# Case when alpha!=1/tauj[j]:
			value += Aj[j]/(alpha-1./tauj[j])*heaviside(deltatj)*( np.exp(-deltatj/tauj[j] )-np.exp(-alpha*deltatj) )
		else:
			# Special case when alpha=1/tauj[j]:
			value += Aj[j]*heaviside(deltatj)*deltatj*np.exp(-alpha*deltatj)
	# Add the zero-th case
	value += + sigma_gas_0*np.exp(-alpha*t)*heaviside(t)

	# Multiply by nuL
	value = nuL*value;
	return(value)
# -------------------------# End of psi -------------------------------------------------



# ipsi ------------------------------------------------------------------------------------
def Get_ipsi(t, chemdict):
	'''Get_ipsi(t, chemdict)
	Returns a primitive of psi(t) function evaluated at time t'''
	assert(0), "Add the tauj=1/alpha case!!!!"
	# Extract the values of the parameters:
	omega = chemdict["omega"];
	R = chemdict["R"];
	nuL = chemdict["nuL"];
	tauj = chemdict["tauj"]
	tj = chemdict["tj"];
	assert(len(tauj)==len(tj)), "tj, tauj lengths mismatch"
	N = len(tj)
	Aj = chemdict["Aj"]
	assert(len(tauj)==len(Aj)), "Aj, tauj lengths mismatch"
	sigma_gas_0 = chemdict["sigma_gas_0"]

	# Useful definitions
	alpha = (1.+omega-R)*nuL
	inv_alpha = 1./alpha; # The inverse of alpha

	# Now separate the cases tauj!=1/alpha and tauj==1/alpha:
	(tauj, tj, Aj, tj_gorro, Aj_gorro) = _separate_cases(alpha, tauj, tj, Aj)

	N, N_gorro = len(tj), len(tj_gorro)
	Kj = [nuL*Aj[j]/(alpha-1./tauj[j]) for j in range(N)]

	value = 0.;
	# Case when alpha!=1/tauj[j]:

	# 3) Integration of Psi (alpha!=1/tauj[j]):
	IntPsi = 0.;
	for j in range(N):
		deltatj = t-tj[j]
		# Positive j:
		IntPsi += Kj[j]*heaviside(deltatj)*tauj[j]*(1.-np.exp(-deltatj/tauj[j]))
		# Negative j:
		IntPsi -= Kj[j]*heaviside(deltatj)/alpha*(1.-np.exp(-deltatj*alpha))
	# "Zero" term:
	IntPsi += nuL*sigma_gas_0*heaviside(t)/alpha*(1.-np.exp(-t*alpha))

	# 4) Integration of Psi (alpha!=1/tauj[j]):
	for j in range(N_gorro):
		deltatj = t-tj_gorro[j]
		# Unique term:
		IntPsi += nuL*Aj_gorro[j]*heaviside(deltatj)*(inv_alpha**2)*(1.-(1.+alpha*deltatj)*np.exp(-alpha*deltatj))

	return(IntPsi)

# -------------------------# End of ipsi -------------------------------------------------




# Get_DTD ------------------------------------------------------------------------------------
def Get_DTD(t, chemdict):
	''' Returns the DTD evaluated at time t
	y = DTD(t, chemdict).
	Deprecated. Use Get_DTD_arr'''
	DTD_value = Get_DTD_arr(t, chemdict)
	return(DTD_value)
# -------------------------# End of DTD -------------------------------------------------




# Get_DTD_arr ------------------------------------------------------------------------------------
def Get_DTD_arr(t, chemdict):
	''' Returns the DTD evaluated at time t
	y = Get_DTD_arr(t, chemdict)'''

	chemdict = prepare_chemdict( chemdict )

	#-------------------------------------
	# Gaussian DTD parameters
	AG_arr = chemdict.get("AG",[])
	Ng = len(AG_arr)# Get the number of Gaussian
	sigma_p_arr = chemdict.get("sigma_p",[])
	taup_arr = chemdict.get("taup",[])
	tau1G_arr = chemdict.get("tau1G",[])
	tau2G_arr = chemdict.get("tau2G",[])

	# Exponential DTD parameters
	AE_arr = chemdict.get("AE",[])
	Ne = len(AE_arr)# Get the number of Exponentials
	tauD_arr = chemdict.get("tauD",[])
	tau1E_arr = chemdict.get("tau1E",[])
	tau2E_arr = chemdict.get("tau2E",[])

	# Inverse DTD parameters
	AI_arr = chemdict.get("AI",[])
	Ni = len(AI_arr)# Get the number of 1/t DTD terms
	tauI_arr = chemdict.get("tauI", Ni*[1.]) # Since it is degenerated with Ai, we set it to one by default
	tau0_arr = chemdict["tau0"];
	tau1I_arr = chemdict.get("tau1I",[])
	tau2I_arr = chemdict.get("tau2I",[])
	#-------------------------------------
	# Safety checks:
	assert( (len(sigma_p_arr)==Ng) and  (len(taup_arr)==Ng) and (len(tau1G_arr)==Ng) and (len(tau2G_arr)==Ng)), "Dimensions of Gaussian DTD parameters mismatch"
	assert( (len(tauD_arr)==Ne) and  (len(tau1E_arr)==Ne) and (len(tau2E_arr)==Ne) ), "Dimensions of Exponential DTD parameters mismatch"
	assert( (len(tauI_arr)==Ni) and  (len(tau0_arr)==Ni) and  (len(tau1I_arr)==Ni) and (len(tau2I_arr)==Ni) ), "Dimensions of 1/t DTD parameters mismatch"
	#-------------------------------------


	# ------------------------------------
	DTD_value = np.zeros_like(t) # Cumulative value

	# The Gaussian term
	for i in range(Ng):
		AG = AG_arr[i]
		if AG==0: continue
		# Extract the DTD parameters:
		sigma_p = sigma_p_arr[i]
		taup = taup_arr[i]
		tau1 = tau1G_arr[i]
		tau2 = tau2G_arr[i]
		DTD_value += AG*np.exp(-0.5*((t-taup)/sigma_p)**2 )*(t>=tau1)*(t<tau2);

	# The Exponential term
	for i in range(Ne):
		AE = AE_arr[i]
		if AE==0: continue
		# Extract the DTD parameters:
		tauD = tauD_arr[i]
		tau1 = tau1E_arr[i]
		tau2 = tau2E_arr[i]
		DTD_value += AE*np.exp(-t/tauD)*(t>=tau1)*(t<tau2);

	# The 1/(t-tau0) term
	for i in range(Ni):
		AI = AI_arr[i]
		if AI==0: continue
		# Extract the DTD parameters:
		tauI = tauI_arr[i]
		tau0 = tau0_arr[i]
		tau1 = tau1I_arr[i]
		tau2 = tau2I_arr[i]
		assert(tau0<tau1), "   ERROR: tau0 must be LOWER than tau1"
		DTD_value += AI*(tauI/((t-tau0)*(t>tau0)+1.*(t<=tau0)))*(t!=tau0)*(t>=tau1)*(t<tau2);
	
	return(DTD_value)
# -------------------------# End of Get_DTD_arr -------------------------------------------------



# Get_infall--------------------------------
def Get_infall(t, chemdict):
	''' Computes the infall given by the t, Aj, and tauj parameters'''
	t = np.array(t);
	I = 0.*t;

	# Read the infall parameters:
	tauj = chemdict.get("tauj", None)
	tj = chemdict.get("tj", None)
	Aj = chemdict.get("Aj", None)

	assert(len(tj)==len(Aj)),"len(tj) is not equal to len(Aj)"
	assert(len(tj)==len(tauj)),"len(tj) is not equal to len(tauj)"
	
	for j in range(len(Aj)): I = Aj[j]*np.exp(-(t-tj[j])*heaviside(t-tj[j])/tauj[j])*heaviside(t-tj[j]);	

	return(I);
# -------------------------# End of Get_infall -------------------------------------------------


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# 				 R1a ASSOCIATED FUNCTIONS
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# R1a_numeric------------------------
def R1a_numeric(t, chemdict):
	''' Numeric solution for R1a(t)
	R1a_numeric(t, chemdict) '''

	chemdict = prepare_chemdict( chemdict )

	# Extract the values of the parameters:
	omega = chemdict["omega"];
	R = chemdict["R"];
	nuL = chemdict["nuL"];
	tauj = chemdict["tauj"]
	tj = chemdict["tj"];
	assert(len(tauj)==len(tj)), "tj, tauj lengths mismatch"
	Ninfall = len(tj)
	Aj = chemdict["Aj"]
	sigma_gas_0 = chemdict["sigma_gas_0"]
	# DTD
	CIa = chemdict["CIa"]
	# Gaussian DTD parameters
	AG_arr = chemdict["AG"]
	Ng = len(AG_arr)# Get the number of Gaussians
	sigma_p_arr = chemdict["sigma_p"]
	taup_arr = chemdict["taup"]
	tau1G_arr = chemdict["tau1G"]
	tau2G_arr = chemdict["tau2G"]
	# Exponential DTD parameters
	AE_arr = chemdict["AE"]
	Ne = len(AE_arr)# Get the number of Exponentials
	tauD_arr = chemdict["tauD"]
	tau1E_arr = chemdict["tau1E"]
	tau2E_arr = chemdict["tau2E"]
	# Inverse DTD parameters
	AI_arr = chemdict["AI"]
	Ni = len(AI_arr)# Get the number of Inverses
	tauI_arr = chemdict["tauI"]
	tau0_arr = chemdict["tau0"]
	tau1I_arr = chemdict["tau1I"]
	tau2I_arr = chemdict["tau2I"]
	#-------------------------------------

	#-------------------------------------
	# Safety checks:
	assert(len(tauj)==Ninfall), "tj, tauj lengths mismatch"
	assert(len(Aj)==Ninfall), "tj, Aj lengths mismatch"
	assert( (len(sigma_p_arr)==Ng) and  (len(taup_arr)==Ng) and (len(tau1G_arr)==Ng) and (len(tau2G_arr)==Ng)), "Dimensions of Gaussian DTD parameters mismatch"
	assert( (len(tauD_arr)==Ne) and  (len(tau1E_arr)==Ne) and (len(tau2E_arr)==Ne) ), "Dimensions of Exponential DTD parameters mismatch"
	assert( (len(tauI_arr)==Ni) and  (len(tau0_arr)==Ni) and  (len(tau1I_arr)==Ni) and (len(tau2I_arr)==Ni) ), "Dimensions of 1/t DTD parameters mismatch"
	#-------------------------------------

	# Useful definitions
	Ninfall = len(tj)# Number of infalls
	alpha = (1.+omega-R)*nuL
	betaj = [alpha - 1./tauj[j] for j in range(Ninfall)]

	# Load the weights and the nodes
	weights, Gauss_nodes = GaussWeightsAndNodes()

	# We want t to be an array:
	try:
		len(t)
	except:
		t = [t];

	#-------------------------------------
	# Numerical integration of the R1a term
	Integr = np.zeros(len(t), dtype=np.float32);

	for t_n, t_now in enumerate(t):
		tau_nodes = 0.5*t_now*(1.+Gauss_nodes); # Nodes of integration
		DTD = Get_DTD_arr(tau_nodes, chemdict) # Perfomed over all the i-th individual DTDs
		psi_t_tau = 0.*tau_nodes
		for j in range(Ninfall):
			deltatj = t_now-tj[j]
			if betaj[j]!=0:
				# The most common case
				inv_betaj = 1./betaj[j];
				psi_t_tau += Aj[j]*inv_betaj*heaviside(deltatj-tau_nodes)*( np.exp(-(deltatj-tau_nodes)/tauj[j] )-np.exp(-alpha*(deltatj-tau_nodes)) )
			else:
				# The peculiar case when alpha=1/tauj[j]
				psi_t_tau += Aj[j]*(deltatj-tau_nodes)*heaviside(deltatj-tau_nodes)*np.exp(-alpha*(deltatj-tau_nodes))
		# Zero term:
		psi_t_tau += sigma_gas_0*heaviside(t_now-tau_nodes)*np.exp(-alpha*(t_now-tau_nodes))
		Integr_tau = DTD*psi_t_tau*(t_now>0);
			
		Integr[t_n] = np.dot(weights, Integr_tau)*(0.5*t_now);

	# Multiply by nuL and CIa:
	Integr = CIa*nuL*Integr;

	return(Integr)
# ---------------------------------------------------------------



# R1a_analytic_gaussian------------------------
def R1a_analytic_gaussian(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, CIa, AG, taup, sigma_p, tau1, tau2):
	''' Exact solution for R1a(t) using Gaussian DTD
	R1a_analytic_gaussian(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, CIa, AG, taup, sigma_p, tau1, tau2) '''

	# Extract the values of the parameters:
	N = len(tj)

	#-------------------------------------
	# Safety checks:
	assert(len(tauj)==len(tj)), "tj, tauj lengths mismatch"
	assert(len(Aj)==len(tj)), "tj, Aj lengths mismatch"
	assert(tau1<tau2),"    Error: tau2 must be GREATER than tau1"
	#-------------------------------------

	# Useful definitions
	Ninfall = len(tj)# Number of infalls
	inv_alpha = 1./alpha;
	betaj = [alpha - 1./tauj[j] for j in range(Ninfall)]	

	# 1) R1a Gaussian term
	# ---------
	R1a_g = 0.

	# Useful definitions
	etaalpha = taup + sigma_p**2*alpha

	# Useful param.
	R1a_gj = np.zeros(N+1, dtype=np.float32).tolist();
	for j in range(Ninfall):
		deltatj = t - tj[j] 
		mint1 = deltatj*(deltatj<tau1) + tau1*(deltatj>=tau1)
		mint2 = deltatj*(deltatj<tau2) + tau2*(deltatj>=tau2)
		if(betaj[j]!=0):
			# Useful definitions
			etaj = taup + sigma_p**2/tauj[j]
			Kj_gorro = Aj[j]/betaj[j]*np.exp(taup/tauj[j] + 0.5*sigma_p**2/tauj[j]**2)
			Kjalpha_gorro = Aj[j]/betaj[j]*np.exp(taup*alpha + 0.5*sigma_p**2*alpha**2)

			# Positive j:
			R1a_gj[j] += Kj_gorro*np.sqrt(np.pi*0.5)*np.exp(-deltatj/tauj[j])*(erf( (mint2-etaj)/np.sqrt(2.)/sigma_p )-erf( (mint1-etaj)/np.sqrt(2.)/sigma_p ))*heaviside(deltatj)
			# Negative j:
			R1a_gj[j] -= Kjalpha_gorro*np.sqrt(np.pi*0.5)*np.exp(-alpha*deltatj)*(erf( (mint2-etaalpha)/np.sqrt(2.)/sigma_p )-erf( (mint1-etaalpha)/np.sqrt(2.)/sigma_p ))*heaviside(deltatj)
		else:
			# Unique term:
			R1a_gj[j] += Aj[j]*np.exp(-alpha*(deltatj-taup-0.5*alpha*sigma_p**2))*(np.sqrt(0.5*np.pi)*(deltatj-etaalpha)*(erf((mint2-etaalpha)/(np.sqrt(2)*sigma_p))-erf((mint1-etaalpha)/(np.sqrt(2)*sigma_p))) + sigma_p*(np.exp(-0.5*((mint2-etaalpha)/sigma_p)**2)-np.exp(-0.5*((mint1-etaalpha)/sigma_p)**2) ) )*heaviside(deltatj)
			print("  Unique term used in the Gaussian DTD")


	# "Zero" term:
	mint1 = t*(t<tau1) + tau1*(t>=tau1)
	mint2 = t*(t<tau2) + tau2*(t>=tau2)
	R1a_gj[-1] = sigma_gas_0*np.exp(-alpha*t)*np.sqrt(np.pi*0.5)*np.exp(taup*alpha+0.5*sigma_p**2*alpha**2)*(erf( (mint2-etaalpha)/np.sqrt(2.)/sigma_p )-erf( (mint1-etaalpha)/np.sqrt(2.)/sigma_p ))*heaviside(t)


	# Constant factor (including nuL)
	for j in range(Ninfall+1): R1a_gj[j] = CIa*AG*sigma_p*nuL*R1a_gj[j];
	R1a_g += np.array(R1a_gj).sum(0); # Sumation
	del(R1a_gj)

	return(R1a_g)
# -------------------------# End of R1a_analytic_gaussian -------------------------------------------------	



# R1a_analytic_exponential------------------------
def R1a_analytic_exponential(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, CIa, AE, tauD, tau1, tau2):
	''' Exact solution for R1a(t) using the Exponential DTD
	R1a_analytic_exponential(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, CIa, AE, tauD, tau1, tau2) '''

	# Extract the values of the parameters:
	N = len(tj)

	#-------------------------------------
	# Safety checks:
	assert(len(tauj)==len(tj)), "tj, tauj lengths mismatch"
	assert(len(Aj)==len(tj)), "tj, Aj lengths mismatch"
	assert(tau1<tau2),"    Error: tau2 must be GREATER than tau1"
	#-------------------------------------

	# Useful definitions
	Ninfall = len(tj)# Number of infalls
	betaj = [alpha - 1./tauj[j] for j in range(Ninfall)]
	inv_alpha = 1./alpha;

	# 1) R1a Exponential term
	# ---------
	R1a_e = 0.

	R1a_ej = np.zeros(N+1, dtype=np.float32).tolist();
	for j in range(Ninfall):
		deltatj = t - tj[j] 
		mint1 = deltatj*(deltatj<tau1) + tau1*(deltatj>=tau1)
		mint2 = deltatj*(deltatj<tau2) + tau2*(deltatj>=tau2)
		if betaj[j]!=0:
			# Qj will discern the special case with taueff-> inf
			# Positive j:
			R1a_ej[j] += Aj[j]/betaj[j]*(QD(mint1, tauj[j], tauD)-QD(mint2, tauj[j], tauD))*np.exp(-deltatj/tauj[j])*heaviside(deltatj)
			# Negative j:
			R1a_ej[j] -= Aj[j]/betaj[j]*(QD(mint1, inv_alpha, tauD)-QD(mint2, inv_alpha, tauD))*np.exp(-deltatj*alpha)*heaviside(deltatj)
		else:
			# Unique term
			R1a_ej[j] += Aj[j]*np.exp(-alpha*deltatj)*( QD_gorro(mint2, deltatj, inv_alpha, tauD)-QD_gorro(mint1, deltatj, inv_alpha, tauD) )
			print("  Unique term used in the Exponential DTD")

	# "Zero" term:
	mint1 = t*(t<tau1) + tau1*(t>=tau1)
	mint2 = t*(t<tau2) + tau2*(t>=tau2)
	R1a_ej[-1] += sigma_gas_0*np.exp(-t*alpha)*( QD(mint1, inv_alpha, tauD)-QD(mint2, inv_alpha, tauD) )*heaviside(t)

	# Constant factor
	for j in range(Ninfall+1): R1a_ej[j] = CIa*AE*nuL*R1a_ej[j];# Including nuL
	R1a_e += np.array(R1a_ej).sum(0); # Sumation
	del(R1a_ej)

	return(R1a_e)
# -------------------------# End of R1a_analytic_exponential -------------------------------------------------



# R1a_analytic_inverse------------------------
def R1a_analytic_inverse(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, CIa, AI, tauI, tau0, tau1, tau2):
	''' Exact solution for R1a(t) using the Inverse of t DTD
	R1a_analytic_inverse(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, CIa, AI, tauI, tau0, tau1, tau2) '''

	# Extract the values of the parameters:
	N = len(tj)

	#-------------------------------------
	# Safety checks:
	assert(len(tauj)==len(tj)), "tj, tauj lengths mismatch"
	assert(len(Aj)==len(tj)), "tj, Aj lengths mismatch"
	assert(tau1<tau2),"    Error: tau2 must be GREATER than tau1"
	assert(tau0<tau1),"    Error: tau0 must be LOWER than tau1"
	#-------------------------------------

	# Useful definitions
	Ninfall = len(tj)# Number of infalls
	inv_alpha = 1./alpha;
	betaj = [alpha - 1./tauj[j] for j in range(Ninfall)]

	# 1) R1a Inverse term
	# ---------
	R1a_i = 0.

	# Offset correction:
	t = t - tau0;
	tau1 = tau1 - tau0;
	tau2 = tau2 - tau0;

	R1a_ij = np.zeros(N+1, dtype=np.float32).tolist();
	for j in range(Ninfall):
		deltatj = t - tj[j]
		mint1 = deltatj*(deltatj<tau1) + tau1*(deltatj>=tau1)
		mint2 = deltatj*(deltatj<tau2) + tau2*(deltatj>=tau2)
		if betaj[j]!=0:
			# Positive j:
			R1a_ij[j] += Aj[j]/betaj[j]*heaviside(deltatj-tau1)*(np.exp(-deltatj/tauj[j])*_safeexpi(mint2/tauj[j])-np.exp(-alpha*deltatj)*_safeexpi(alpha*mint2))
			# Negative j:
			R1a_ij[j] -= Aj[j]/betaj[j]*heaviside(deltatj-tau1)*(np.exp(-deltatj/tauj[j])*_safeexpi(tau1/tauj[j])-np.exp(-alpha*deltatj)*_safeexpi(alpha*tau1))
		else:
			# Unique term:
			R1a_ij[j] += Aj[j]*( deltatj*np.exp(-alpha*deltatj)*(_safeexpi(alpha*mint2)-_safeexpi(alpha*mint1)) + inv_alpha*np.exp(alpha*(mint1-deltatj)) - inv_alpha*np.exp(alpha*(mint2-deltatj))   )
			print("  Unique term used in the inverse DTD")

	# "Zero" term:
	mint1 = t*(t<tau1) + tau1*(t>=tau1)
	mint2 = t*(t<tau2) + tau2*(t>=tau2)
	R1a_ij[-1] += sigma_gas_0*np.exp(-alpha*t)*(_safeexpi(mint2*alpha)-_safeexpi(mint1*alpha))*heaviside(t)

	# Constant factor
	for j in range(Ninfall+1): R1a_ij[j] = CIa*tauI*AI*nuL*R1a_ij[j];# Includes the nuL term
	R1a_i += np.array(R1a_ij).sum(0); # Sumation
	del(R1a_ij)

	return(R1a_i)
# -------------------------# End of R1a_analytic_inverse -------------------------------------------------




# R1a_analytic------------------------
def R1a_analytic(t, chemdict, separated_terms=False):
	''' Exact solution for R1a(t)
	R1a_analytic(t, chemdict) 
	When separated_terms is True, the output is a 3-element array'''

	chemdict = prepare_chemdict( chemdict )

	# Extract the values of the parameters:
	omega = chemdict["omega"];
	R = chemdict["R"];
	nuL = chemdict["nuL"];
	tauj = chemdict["tauj"]
	tj = chemdict["tj"];
	assert(len(tauj)==len(tj)), "tj, tauj lengths mismatch"
	N = len(tj)
	Aj = chemdict["Aj"]
	sigma_gas_0 = chemdict["sigma_gas_0"]
	# DTD
	CIa = chemdict["CIa"]
	#-------------------------------------


	#-------------------------------------
	# Safety checks:
	assert(len(tauj)==len(tj)), "tj, tauj lengths mismatch"
	assert(len(Aj)==len(tj)), "tj, Aj lengths mismatch"
	#-------------------------------------

	# Useful definitions
	alpha = (1.+omega-R)*nuL
	gamma = nuL*(1.+omega-R);
	

	# 1) R1a Gaussian term
	# ---------
	# Gaussian DTD parameters
	AG_arr = chemdict["AG"]
	Ng = len(AG_arr)# Get the number of Gaussians
	sigma_p_arr = chemdict["sigma_p"]
	taup_arr = chemdict["taup"]
	tau1G_arr = chemdict["tau1G"]
	tau2G_arr = chemdict["tau2G"]

	assert( (len(sigma_p_arr)==Ng) and  (len(taup_arr)==Ng) and (len(tau1G_arr)==Ng) and (len(tau2G_arr)==Ng)), "Dimensions of Gaussian DTD parameters mismatch"

	R1a_g = 0. # Cumulative value
	for i in range(Ng):
		AG = AG_arr[i]
		taup = taup_arr[i]
		sigma_p = sigma_p_arr[i]
		tau1 = tau1G_arr[i]
		tau2 = tau2G_arr[i]

		R1a_g += R1a_analytic_gaussian(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, CIa, AG, taup, sigma_p, tau1, tau2)


	# 2) R1a Exponential term
	# ---------
	# Exponential DTD parameters
	AE_arr = chemdict["AE"]
	Ne = len(AE_arr)# Get the number of Exponentials
	tauD_arr = chemdict["tauD"]
	tau1E_arr = chemdict["tau1E"]
	tau2E_arr = chemdict["tau2E"]

	assert( (len(tauD_arr)==Ne) and  (len(tau1E_arr)==Ne) and (len(tau2E_arr)==Ne) ), "Dimensions of Exponential DTD parameters mismatch"

	R1a_e = 0.# Cumulative value
	for i in range(Ne):
		AE = AE_arr[i]
		tauD = tauD_arr[i]
		tau1 = tau1E_arr[i]
		tau2 = tau2E_arr[i]

		R1a_e += R1a_analytic_exponential(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, CIa, AE, tauD, tau1, tau2);


	# 3) R1a Inverse term
	# ---------
	# Inverse DTD parameters
	AI_arr = chemdict["AI"]
	Ni = len(AI_arr)# Get the number of Inverses
	tauI_arr = chemdict["tauI"]
	tau0_arr = chemdict["tau0"]
	tau1I_arr = chemdict["tau1I"]
	tau2I_arr = chemdict["tau2I"]

	assert( (len(tauI_arr)==Ni) and  (len(tau0_arr)==Ni) and  (len(tau1I_arr)==Ni) and (len(tau2I_arr)==Ni) ), "Dimensions of 1/t DTD parameters mismatch"

	R1a_i = 0.# Cumulative value
	for i in range(Ni):
		AI = AI_arr[i]
		tauI = tauI_arr[i]
		tau0 = tau0_arr[i]
		tau1 = tau1I_arr[i]
		tau2 = tau2I_arr[i]

		R1a_i += R1a_analytic_inverse(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, CIa, AI, tauI, tau0, tau1, tau2)

	# Prepare the output
	if separated_terms:
		return([R1a_g,R1a_e,R1a_i])
	else:
		return(R1a_g+R1a_e+R1a_i)
# -------------------------# End of R1a_analytic -------------------------------------------------




# iR1a_analytic----------------------------------------------------------------------------------
def iR1a_analytic(t, chemdict):
	chemdict = prepare_chemdict( chemdict )

	# Extract the values of the parameters:
	omega = chemdict["omega"];
	R = chemdict["R"];
	nuL = chemdict["nuL"];
	tauj = chemdict["tauj"]
	tj = chemdict["tj"];
	assert(len(tauj)==len(tj)), "tj, tauj lengths mismatch"
	N = len(tj)
	Aj = chemdict["Aj"]
	sigma_gas_0 = chemdict["sigma_gas_0"]
	# DTD
	CIa = chemdict["CIa"]
	# Gaussian DTD parameters
	AG_arr = chemdict["AG"]
	Ng = len(AG_arr)# Get the number of Gaussians
	sigma_p_arr = chemdict["sigma_p"]
	taup_arr = chemdict["taup"]
	tau1G_arr = chemdict["tau1G"]
	tau2G_arr = chemdict["tau2G"]
	# Exponential DTD parameters
	AE_arr = chemdict["AE"]
	Ne = len(AE_arr)# Get the number of Exponentials
	tauD_arr = chemdict["tauD"]
	tau1E_arr = chemdict["tau1E"]
	tau2E_arr = chemdict["tau2E"]
	# Inverse DTD parameters
	AI_arr = chemdict["AI"]
	Ni = len(AI_arr)# Get the number of Inverses
	tauI_arr = chemdict["tauI"]
	tau0_arr = chemdict["tau0"]
	tau1I_arr = chemdict["tau1I"]
	tau2I_arr = chemdict["tau2I"]
	#-------------------------------------

	#-------------------------------------
	# Safety checks:
	assert(len(tauj)==len(tj)), "tj, tauj lengths mismatch"
	assert(len(Aj)==len(tj)), "tj, Aj lengths mismatch"
	assert( (len(sigma_p_arr)==Ng) and  (len(taup_arr)==Ng) and (len(tau1G_arr)==Ng) and (len(tau2G_arr)==Ng)), "Dimensions of Gaussian DTD parameters mismatch"
	assert( (len(tauD_arr)==Ne) and  (len(tau1E_arr)==Ne) and (len(tau2E_arr)==Ne) ), "Dimensions of Exponential DTD parameters mismatch"
	assert( (len(tauI_arr)==Ni) and  (len(tau0_arr)==Ni) and  (len(tau1I_arr)==Ni) and (len(tau2I_arr)==Ni) ), "Dimensions of 1/t DTD parameters mismatch"
	#-------------------------------------

	# Useful definitions
	Ninfall = len(tj)# Number of infalls
	alpha = (1.+omega-R)*nuL
	betaj = [- 1./tauj[j] for j in range(Ninfall)]
	Kj = [nuL*Aj[j]/(alpha-1./tauj[j]) for j in range(Ninfall)]

	# ------------------------------------------------------------------------
	# 1) Non-Homogeneous non-trivial gaussian term:
	global_nhg = 0.;
	for i in range(Ng):
		AG = AG_arr[i]
		taup = taup_arr[i]
		sigma_p = sigma_p_arr[i]
		tau1 = tau1G_arr[i]
		tau2 = tau2G_arr[i]

		# Useful definitions
		etaj = [(taup+sigma_p**2/tauj[j])  for j in range(Ninfall)]
		Kj_gorro = [Kj[j]*np.exp(taup/tauj[j] + 0.5*sigma_p**2/tauj[j]**2) for j in range(Ninfall)]
		Kjalpha_gorro = [Kj[j]*np.exp(taup*alpha + 0.5*sigma_p**2*alpha**2) for j in range(Ninfall)]
		etaalpha = taup + sigma_p**2*alpha

		aux_nhg = 0.;
		for j in range(Ninfall):
			deltatj = t - tj[j];
			mint1 = deltatj*(deltatj<tau1) + tau1*(deltatj>=tau1)
			mint2 = deltatj*(deltatj<tau2) + tau2*(deltatj>=tau2)

			# From 0 to tau2:
			aux_nhg += -Kj_gorro[j]*heaviside(deltatj-tau1)*tauj[j]*np.exp(betaj[j]*etaj[j]+0.5*betaj[j]**2*sigma_p**2)*(erf( (taup-mint2)/(np.sqrt(2.)*sigma_p))-erf( (taup-mint1)/(np.sqrt(2.)*sigma_p))); # Positive (first term)
			aux_nhg += -Kj_gorro[j]*heaviside(deltatj-tau1)*tauj[j]*np.exp(betaj[j]*mint2)*(erf( (mint2-etaj[j])/(np.sqrt(2.)*sigma_p)   )-erf( (tau1-etaj[j])/(np.sqrt(2.)*sigma_p)   ))# Positive (second term)

			aux_nhg += Kjalpha_gorro[j]*heaviside(deltatj-tau1)/alpha*np.exp(-alpha*etaalpha-0.5*alpha**2*sigma_p**2)*(erf( (-alpha*sigma_p**2+etaalpha-mint2)/(np.sqrt(2.)*sigma_p))-erf( (-alpha*sigma_p**2+etaalpha-mint1)/(np.sqrt(2.)*sigma_p))); # Negative (first term)
			aux_nhg += Kjalpha_gorro[j]*heaviside(deltatj-tau1)/alpha*np.exp(-alpha*mint2)*(erf( (mint2-etaalpha)/(np.sqrt(2.)*sigma_p)   )-erf( (tau1-etaalpha)/(np.sqrt(2.)*sigma_p)   ))# Negative (second term)
		# Zero term:
		mint1 = t*(t<tau1) + tau1*(t>=tau1)
		mint2 = t*(t<tau2) + tau2*(t>=tau2)
		aux_nhg += -nuL*sigma_gas_0*heaviside(t-tau1)/alpha*np.exp(-alpha*etaalpha-0.5*alpha**2*sigma_p**2)*(erf( (-alpha*sigma_p**2+etaalpha-mint2)/(np.sqrt(2.)*sigma_p))-erf( (-alpha*sigma_p**2+etaalpha-mint1)/(np.sqrt(2.)*sigma_p))); # Zero term (first term)
		aux_nhg -= nuL*sigma_gas_0*heaviside(t-tau1)/alpha*np.exp(-alpha*mint2)*(erf( (mint2-etaalpha)/(np.sqrt(2.)*sigma_p)   )-erf( (tau1-etaalpha)/(np.sqrt(2.)*sigma_p)   )) # Zero term (second term)


		# From tau2 to t:
		for j in range(Ninfall):
			deltatj = t - tj[j];
			# From tau2 to t:
			aux_nhg += -Kj_gorro[j]*heaviside(deltatj-tau2)*tauj[j]*( erf( (tau2-etaj[j])/(np.sqrt(2.)*sigma_p)) - erf( (tau1-etaj[j])/(np.sqrt(2.)*sigma_p))  )*(np.exp(betaj[j]*deltatj)-np.exp(betaj[j]*tau2) ); # Positive (unique term)
			aux_nhg += Kjalpha_gorro[j]*heaviside(deltatj-tau2)*alpha*( erf( (tau2-etaalpha)/(np.sqrt(2.)*sigma_p)) - erf( (tau1-etaalpha)/(np.sqrt(2.)*sigma_p))  )*(np.exp(-alpha*deltatj)-np.exp(-alpha*tau2) ) # Negative (unique term)

		# Zero term:
		aux_nhg -= nuL*sigma_gas_0*heaviside(t-tau2)*alpha*( erf( (tau2-etaalpha)/(np.sqrt(2.)*sigma_p)) - erf( (tau1-etaalpha)/(np.sqrt(2.)*sigma_p))  )*(np.exp(-alpha*t)-np.exp(-alpha*tau2) ) # Zero term (unique term)

		global_nhg += np.sqrt(np.pi*0.5)*sigma_p*CIa*AG*aux_nhg# Multiply by the constants
	# ------------------------------------------------------------------------


	# ------------------------------------------------------------------------
	# 2) Non-Homogeneous non-trivial exponential term:
	global_nhe = 0.;
	for i in range(Ne):
		AE = AE_arr[i]
		tauD = tauD_arr[i]
		tau1 = tau1E_arr[i]
		tau2 = tau2E_arr[i]

		# Useful definitions
		taueffj = [ 1./(1./tauD-1./tauj[j]) for j in range(Ninfall)]
		taualpha = 1./(1./tauD-alpha)

		aux_nhe = 0.;
		for j in range(Ninfall):
			deltatj = t - tj[j];
			mint2 = deltatj*(deltatj<tau2) + tau2*(deltatj>=tau2)

			# From 0 to tau2:
			aux_nhe += Kj[j]*taueffj[j]*heaviside(deltatj-tau1)*(   -(np.exp(betaj[j]*mint2-tau1/taueffj[j])-np.exp(betaj[j]*tau1-tau1/taueffj[j]))*tauj[j] + tauD*(np.exp(-mint2/tauD)-np.exp(-tau1/tauD) )  )# Positive term
			aux_nhe -= Kj[j]*taualpha*heaviside(deltatj-tau1)*(   (np.exp(-alpha*mint2-tau1/taualpha)-np.exp(-tau1/tauD))/(-alpha) + tauD*(np.exp(-mint2/tauD)-np.exp(-tau1/tauD) )  )# Negative term


		# Zero term:
		mint2 = t*(t<tau2) + tau2*(t>=tau2)
		aux_nhe += nuL*sigma_gas_0*taualpha*heaviside(t-tau1)*(   (np.exp(-alpha*mint2-tau1/taualpha)-np.exp(-tau1/tauD))/(-alpha) + tauD*(np.exp(-mint2/tauD)-np.exp(-tau1/tauD) )  )# Zero term


		# From tau2 to t:
		for j in range(Ninfall):
			deltatj = t - tj[j];

			aux_nhe += -Kj[j]*taueffj[j]*heaviside(deltatj-tau2)*tauj[j]*(np.exp(-tau1/taueffj[j])-np.exp(-tau2/taueffj[j]))*(np.exp(-deltatj/tauj[j])-np.exp(-tau2/tauj[j])); # Positive (unique term)

			aux_nhe -= -Kj[j]*taualpha*heaviside(deltatj-tau2)/alpha*(np.exp(-tau1/taualpha)-np.exp(-tau2/taualpha))*(np.exp(-alpha*deltatj)-np.exp(-alpha*tau2)) # Negative (unique term)

		# Zero term:
		aux_nhe += -nuL*sigma_gas_0*taualpha*heaviside(t-tau2)/alpha*(np.exp(-tau1/taualpha)-np.exp(-tau2/taualpha))*(np.exp(-alpha*t)-np.exp(-alpha*tau2)) # Zero term (unique term)

		global_nhe += CIa*AE*aux_nhe# Multiply by the constants
	# ------------------------------------------------------------------------


	# ------------------------------------------------------------------------
	# 3) Non-Homogeneous non-trivial inverse term:
	global_nhi = 0.;
	for i in range(Ni):
		AI = AI_arr[i]
		tauI = tauI_arr[i]
		tau0 = tau0_arr[i]
		tau1 = tau1I_arr[i] - tau0; # We apply the offset to tau1 here
		tau2 = tau2I_arr[i] - tau0; # We apply the offset to tau2 here
		assert(tau1>0),"   ERROR: tau0 must be LOWER than tau1 (tau0<tau1)"
		assert(tau1<=tau2),"   ERROR: tau1 must be LOWER than tau2 (tau1<tau2)"

		aux_nhi = 0.;
		for j in range(Ninfall):
			deltatj = t - tj[j] - tau0; # We apply the offset to deltatj here
			mint1 = deltatj*(deltatj<tau1) + tau1*(deltatj>=tau1); # mint1 already has the offset in tau0
			mint2 = deltatj*(deltatj<tau2) + tau2*(deltatj>=tau2) # mint2 already has the offset in tau0

			# From 0 to tau2 and from tau2 to t:
			aux_nhi += Kj[j]*tauj[j]*heaviside(deltatj-tau1)*(_safelog(mint2,mint1)-np.exp(-mint2/tauj[j])*_safeexpi(mint2/tauj[j]) + _safeexpi(tau2/tauj[j])*( np.exp(-mint2/tauj[j])-np.exp(-deltatj/tauj[j]) ) + _safeexpi(tau1/tauj[j])*np.exp(-deltatj/tauj[j]) )# Positive term
			aux_nhi -= Kj[j]/alpha*heaviside(deltatj-tau1)*(_safelog(mint2,mint1)-np.exp(-alpha*mint2)*_safeexpi(alpha*mint2) + _safeexpi(alpha*tau2)*( np.exp(-alpha*mint2)-np.exp(-alpha*deltatj) ) + _safeexpi(alpha*tau1)*np.exp(-alpha*deltatj) )# Negative term
		
		# Zero term:
		t0 = t - tau0; # We apply the offset to t here. No dependence on "t" should be observed in the "zero term" but on "t0"
		del(t)#We do not need it anymore here
		mint1 = t0*(t0<tau1) + tau1*(t0>=tau1)# mint1 already has the offset in tau0
		mint2 = t0*(t0<tau2) + tau2*(t0>=tau2)# mint2 already has the offset in tau0

		aux_nhi += nuL*sigma_gas_0*heaviside(t0-tau1)/alpha*(_safelog(mint2,mint1)-np.exp(-alpha*mint2)*_safeexpi(alpha*mint2) + _safeexpi(alpha*tau2)*( np.exp(-alpha*mint2)-np.exp(-alpha*t0) ) + _safeexpi(alpha*tau1)*np.exp(-alpha*t0) )# Zero term	
	

		global_nhi += CIa*AI*tauI*aux_nhi# Multiply by the constants


	# Combine both terms:
	primitive_R1a = global_nhg + global_nhe + global_nhi

	return(primitive_R1a)
# -------------------------# End of iR1a_analytic -------------------------------------------



# Get_CIa-------------------------------------------------------------------------------------
def Get_CIa(TypeIa_SNe_ratio, Area, chemdict, present_day_time=13.8):
	''' Set the value CIa given a TypeIa_SNe_ratio at present-da and an area:
	Get_CIa(TypeIa_SNe_ratio, Area, chemdict, present_day_time=13.8)

	TypeIa_SNe_ratio in events per Gyr
	Area in pc**2
	chemdict: Chemical dictionary associated with the model.
	present_day_time: Evolutionary time (in Gyr)

	returns: CIa'''

	chemdict["CIa"] = chemdict.get("CIa",1.);
	R1a_today = R1a_analytic(present_day_time, chemdict)
	CIa = TypeIa_SNe_ratio/Area/R1a_today*chemdict.get("CIa",1.);

	return(CIa)
# -------------------------# End of Get_CIa -------------------------------------------------




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# 				 MODEL EQUATION AND SOLVERS
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -




# SolveChemEvolModel_GaussianTerm -------------------------------------------------------------------------------
def SolveChemEvolModel_GaussianTerm(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, CIa, AG, taup, sigma_p, tau1, tau2, mx1a):
	'''
	Returns the part of the solution that corresponds to the Gaussian DTD.
	'''
	assert(len(tauj)==len(tj)), "tj, tauj lengths mismatch"
	N = len(tj)
	assert(tau2>tau1), "tau2<=tau1"
	#-------------------------------------

	#-------------------------------------
	# Useful definitions
	gamma = alpha
	N = len(tj)
	betaj = [gamma - 1./tauj[j] for j in range(N)]
	etaalpha = taup + sigma_p**2*alpha
	#-------------------------------------

	#-------------------------------------
	#problem = [betaj[j]*etaj[j]+betaj[j]**2*sigma_p**2*0.5 for j in range(N)]
	#for j in range(N): print(j, problem[j], np.round(np.exp(problem[j]),3) )
	#print("....")
	#problem = [taup/tauj[j]+0.5*sigma_p**2/tauj[j]**2 for j in range(N)]
	#for j in range(N): print(j, problem[j], np.round(np.exp(problem[j]),3) )
	#del(problem)
	#-------------------------------------

	# ------------------------------------------------------------------------
	# 1) Non-Homogeneous non-trivial gaussian term:
	sol_gauss = 0.;
	for j in range(N):
		# Time variables
		deltatj = t - tj[j];
		mint1 = deltatj*(deltatj<tau1) + tau1*(deltatj>=tau1)
		mint2 = deltatj*(deltatj<tau2) + tau2*(deltatj>=tau2)

		if betaj[j]!=0:
			# Case alpha!=1/tauj[j]

			# Useful parameters
			etaj = taup+sigma_p**2/tauj[j]
			Kj_gorro = Aj[j]/betaj[j]*np.exp(taup/tauj[j] + 0.5*sigma_p**2/tauj[j]**2)
			Kjalpha_gorro = Aj[j]/betaj[j]*np.exp(taup*alpha + 0.5*sigma_p**2*alpha**2)

			# From 0 to tau2:
			sol_gauss += Kj_gorro*heaviside(deltatj-tau1)*np.exp(-gamma*deltatj)/betaj[j]*np.exp(betaj[j]*etaj+0.5*betaj[j]**2*sigma_p**2)*(erf( (betaj[j]*sigma_p**2+etaj-mint2)/(np.sqrt(2.)*sigma_p))-erf( (betaj[j]*sigma_p**2+etaj-mint1)/(np.sqrt(2.)*sigma_p))); # Positive (first term)
			sol_gauss += Kj_gorro*heaviside(deltatj-tau1)*np.exp(-gamma*deltatj)/betaj[j]*np.exp(betaj[j]*mint2)*(erf( (mint2-etaj)/(np.sqrt(2.)*sigma_p)   )-erf( (tau1-etaj)/(np.sqrt(2.)*sigma_p)   ))# Positive (second term)

			sol_gauss -= Kjalpha_gorro*heaviside(deltatj-tau1)*np.exp(-gamma*deltatj)*( (mint2-etaalpha)*erf( (mint2-etaalpha)/(np.sqrt(2.)*sigma_p) )-(mint2-mint1)*erf( (tau1-etaalpha)/(np.sqrt(2.)*sigma_p) )    - (mint1-etaalpha)*erf( (mint1-etaalpha)/(np.sqrt(2.)*sigma_p) )  ); # Negative (first term)
			sol_gauss -= Kjalpha_gorro*heaviside(deltatj-tau1)*np.exp(-gamma*deltatj)*np.sqrt(2./np.pi)*sigma_p*( np.exp(-(mint2-etaalpha)**2/(2.*sigma_p**2) )-np.exp(-(mint1-etaalpha)**2/(2.*sigma_p**2) ) ) # Negative (second term)

			# From tau2 to t:
			sol_gauss += Kj_gorro*heaviside(deltatj-tau2)/betaj[j]*np.exp(-gamma*deltatj)*( erf( (tau2-etaj)/(np.sqrt(2.)*sigma_p)) - erf( (tau1-etaj)/(np.sqrt(2.)*sigma_p))  )*(np.exp(betaj[j]*deltatj)-np.exp(betaj[j]*tau2) ); # Positive (unique term)
			sol_gauss -= Kjalpha_gorro*heaviside(deltatj-tau2)*np.exp(-gamma*deltatj)*(deltatj-tau2)*( erf( (tau2-etaalpha)/(np.sqrt(2.)*sigma_p)) - erf( (tau1-etaalpha)/(np.sqrt(2.)*sigma_p))  ) # Negative (unique term)
		else:
			sol_gauss += Aj[j]*np.exp(-alpha*(deltatj-taup-0.5*alpha*sigma_p**2))*( (deltatj-etaalpha)*(erf( (mint2-etaalpha)/(np.sqrt(2.)*sigma_p ) ) - erf( (mint1-etaalpha)/(np.sqrt(2.)*sigma_p) ) ) + sigma_p*(np.exp(-0.5*((mint2-etaalpha)/sigma_p)**2)-np.exp(-0.5*((mint1-etaalpha)/sigma_p)**2) ) ); # Unique term

	# Zero term. From 0 to tau2:
	mint1 = t*(t<tau1) + tau1*(t>=tau1)
	mint2 = t*(t<tau2) + tau2*(t>=tau2)
	sol_gauss += sigma_gas_0*np.exp(taup*alpha+0.5*sigma_p**2*alpha**2)*heaviside(t-tau1)*np.exp(-gamma*t)*( (mint2-etaalpha)*erf( (mint2-etaalpha)/(np.sqrt(2.)*sigma_p) )-mint2*erf( (tau1-etaalpha)/(np.sqrt(2.)*sigma_p) )    - (mint1-etaalpha)*erf( (mint1-etaalpha)/(np.sqrt(2.)*sigma_p) ) + mint1*erf( (tau1-etaalpha)/(np.sqrt(2.)*sigma_p) )   ); # Zero term (first term)
	sol_gauss += sigma_gas_0*np.exp(taup*alpha+0.5*sigma_p**2*alpha**2)*heaviside(t-tau1)*np.exp(-gamma*t)*np.sqrt(2./np.pi)*sigma_p*( np.exp(-(mint2-etaalpha)**2/(2.*sigma_p**2) )-np.exp(-(mint1-etaalpha)**2/(2.*sigma_p**2) ) ) # Zero term (second term)

	# Zero term. From tau2 to t:
	sol_gauss += sigma_gas_0*np.exp(taup*alpha+0.5*sigma_p**2*alpha**2)*heaviside(t-tau2)*np.exp(-gamma*t)*(t-tau2)*( erf( (tau2-etaalpha)/(np.sqrt(2.)*sigma_p)) - erf( (tau1-etaalpha)/(np.sqrt(2.)*sigma_p))  ) # Zero term (unique term)

	# Combine all the terms
	sol_gauss = mx1a*np.sqrt(np.pi*0.5)*sigma_p*CIa*AG*nuL*sol_gauss# Multiply by the constants (including nuL)
	# ------------------------------------------------------------------------


	return( sol_gauss )
# -------------------------# End of SolveChemEvolModel_GaussianTerm -------------------------------------------------





# SolveChemEvolModel_ExponentialTerm -------------------------------------------------------------------------------
def SolveChemEvolModel_ExponentialTerm(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, CIa, AE, tauD, tau1, tau2, mx1a):
	'''
	Returns the part of the solution that corresponds to the Exponential DTD.
	'''
	assert(len(tauj)==len(tj)), "tj, tauj lengths mismatch"
	N = len(tj)
	assert(tau2>tau1), "tau2<=tau1"
	#-------------------------------------

	#-------------------------------------
	# Useful definitions
	gamma = alpha
	inv_alpha = 1./alpha;
	N = len(tj)
	betaj = [gamma - 1./tauj[j] for j in range(N)]
	#-------------------------------------


	# ------------------------------------------------------------------------
	# 1) Non-Homogeneous non-trivial exponential term:
	sol_expo = 0.;
	for j in range(N):
		# Time variable
		deltatj = t - tj[j];
		mint1 = deltatj*(deltatj<tau1) + tau1*(deltatj>=tau1)
		mint2 = deltatj*(deltatj<tau2) + tau2*(deltatj>=tau2)

		if betaj[j]!=0:
			# Case alpha!=1/tauj[j]
			# From 0 to t:
			##print("Normal case")
			sol_expo += Aj[j]/betaj[j]*np.exp(-gamma*deltatj)*( QD(tau2, tauj[j], tauD)*(QD(deltatj, inv_alpha, tauj[j]) -QD(mint2, inv_alpha, tauj[j]) ) +QD(tau1, tauj[j], tauD)*( QD(mint1, inv_alpha, tauj[j])-QD(deltatj, inv_alpha, tauj[j]) ) + QD(tau2, inv_alpha, tauD)*( QD(mint2, inv_alpha, inv_alpha) - QD(deltatj, inv_alpha, inv_alpha)  )+QD(tau1, inv_alpha, tauD)*( QD(deltatj, inv_alpha, inv_alpha) - QD(mint1, inv_alpha, inv_alpha) )+PD(mint2, tauj[j], inv_alpha, tauD) - SD(mint2, inv_alpha, tauD) - PD(mint1, tauj[j], inv_alpha, tauD) + SD(mint1, inv_alpha, tauD)  );# Positive and negative terms
		else:
			# Case alpha==1/tauj[j]
			sol_expo += Aj[j]*np.exp(-deltatj/tauD)*(SD_gorro(deltatj-mint2, inv_alpha, tauD) - SD_gorro(deltatj-mint1, inv_alpha, tauD) )
	#assert(),"Problemas"
	# Zero term. From 0 to t:
	mint1 = t*(t<tau1) + tau1*(t>=tau1)
	mint2 = t*(t<tau2) + tau2*(t>=tau2)
	sol_expo += sigma_gas_0*np.exp(-gamma*t)*( QD(tau2, inv_alpha, tauD)*( QD(t, inv_alpha, inv_alpha)-QD(mint2, inv_alpha, inv_alpha) )- QD(tau1, inv_alpha, tauD)*( QD(t, inv_alpha, inv_alpha)-QD(mint1, inv_alpha, inv_alpha) ) + SD(mint2, inv_alpha, tauD) - SD(mint1, inv_alpha, tauD) ); # Positive and negative terms

	# Combine all the terms
	sol_expo = mx1a*CIa*AE*nuL*sol_expo; # Multiply by the constants (including nuL)
	# ------------------------------------------------------------------------


	return( sol_expo )
# -------------------------# End of SolveChemEvolModel_ExponentialTerm -------------------------------------------------






# SolveChemEvolModel_InverseTerm -------------------------------------------------------------------------------
def SolveChemEvolModel_InverseTerm(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, CIa, AI, tauI, tau0, tau1, tau2, mx1a):
	'''
	Returns the part of the solution that corresponds to the Inverse DTD.
	'''
	assert(len(tauj)==len(tj)), "tj, tauj lengths mismatch"
	N = len(tj)
	assert(tau2>tau1), "tau2<=tau1"
	#-------------------------------------

	#-------------------------------------
	# Useful definitions
	gamma = alpha
	inv_alpha = 1./alpha;
	N = len(tj)
	betaj = [gamma - 1./tauj[j] for j in range(N)]

	#-------------------------------------

	# tau0 shifts all the deltatj, tau1 and tau2
	tau1_0 = tau1 - tau0;# We should not overwrite tau1 because it is used later
	tau2_0 = tau2 - tau0;# We should not overwrite tau2 because it is used later
	#-------------------------------------


	# ------------------------------------------------------------------------
	# 1) Non-Homogeneous non-trivial inverse term:

	sol_inv = 0.;
	for j in range(N):
		# Time variables
		deltatj = t - tj[j]-tau0;# Shift induced by tau0:
		mint2 = deltatj*(deltatj<tau2_0) + tau2_0*(deltatj>=tau2_0)# Already have the shift in tau0 (because of deltatj, tau2_0)

		if betaj[j]!=0:
			# Case alpha!=1/tauj[j]
			inv_betaj = 1./betaj[j]
			# From 0 to tau2:
			sol_inv += Aj[j]*np.power(inv_betaj, 2.)*heaviside(deltatj-tau1_0)*np.exp(-gamma*deltatj)*(np.exp(betaj[j]*mint2)*_safeexpi(mint2/tauj[j])-_safeexpi(gamma*mint2)+_safeexpi(gamma*tau1_0)-_safeexpi(tau1_0/tauj[j])*np.exp(betaj[j]*deltatj)  )# Positive term
			sol_inv -= Aj[j]*inv_betaj*heaviside(deltatj-tau1_0)*np.exp(-gamma*deltatj)*(mint2*_safeexpi(alpha*mint2)-tau1_0*_safeexpi(alpha*tau1_0) -1./alpha*(np.exp(alpha*mint2)-np.exp(alpha*tau1_0)) -_safeexpi(alpha*tau1_0)*(deltatj-tau1_0) )# Negative term

			# From tau2 to t:
			sol_inv += Aj[j]*np.power(inv_betaj,2.)*heaviside(deltatj-tau2_0)*np.exp(-gamma*deltatj)*_safeexpi(tau2_0/tauj[j])*(np.exp(betaj[j]*deltatj)-np.exp(betaj[j]*tau2_0) ); # Positive (unique term)
			sol_inv -= Aj[j]*inv_betaj*heaviside(deltatj-tau2_0)*np.exp(-gamma*deltatj)*_safeexpi(alpha*tau2_0)*( deltatj-tau2_0 ) # Negative (unique term)
		else:
			# Case alpha==1/tauj[j]
			sol_inv += Aj[j]*np.exp(-alpha*deltatj)*heaviside( deltatj-tau1_0)*(0.5*(mint2**2)*_safeexpi(alpha*mint2) -0.5*(deltatj**2)*_safeexpi(alpha*tau1_0) + 0.5*(inv_alpha**2)*(np.exp(alpha*tau1_0) - np.exp(alpha*mint2) ) + 0.5*inv_alpha*(tau1_0*np.exp(alpha*tau1_0) - mint2*np.exp(alpha*mint2) ) + inv_alpha*(deltatj-tau1_0)*np.exp(alpha*tau1_0) + heaviside(deltatj-tau2_0)*( 0.5*_safeexpi(alpha*tau2_0)*(deltatj**2-tau2_0**2) - inv_alpha*(deltatj-tau2_0)*np.exp(alpha*tau2_0) ))# Unique term

	# Zero term.
	deltatj = t - tau0;
	mint2 = deltatj*(deltatj<tau2_0) + tau2_0*(deltatj>=tau2_0)
	#	From 0 to tau2
	sol_inv += sigma_gas_0*heaviside(deltatj-tau1_0)*np.exp(-gamma*deltatj)*(mint2*_safeexpi(alpha*mint2)-1./alpha*(np.exp(alpha*mint2)-np.exp(alpha*tau1_0)) -_safeexpi(alpha*tau1_0)*deltatj )# Zero term

	# 	From tau2 to t:
	sol_inv += sigma_gas_0*heaviside(deltatj-tau2_0)*np.exp(-gamma*deltatj)*_safeexpi(alpha*tau2_0)*( deltatj-tau2_0 ) # Zero term (unique term)
	
	sol_inv = mx1a*CIa*AI*tauI*nuL*sol_inv# Multiply by the constants (including nuL)
	# ------------------------------------------------------------------------

	return( sol_inv )
# -------------------------# End of SolveChemEvolModel_InverseTerm -------------------------------------------------



# SolveChemEvolModel_InhomogeneousTrivialTerm --------------------------------------------------------------------------------------------------
def SolveChemEvolModel_InhomogeneousTrivialTerm(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, yx, R):
	'''
	Returns the part of the solution that corresponds to the Inhomogeneous Trivial term.
	'''
	assert(len(tauj)==len(tj)), "tj, tauj lengths mismatch"
	N = len(tj)
	#-------------------------------------

	#-------------------------------------
	# Useful definitions
	gamma = alpha
	N = len(tj)
	betaj = [gamma - 1./tauj[j] for j in range(N)]
	#-------------------------------------


	# ------------------------------------------------------------------------
	# 2) Non-Homogeneous trivial term:
	sol_nht = 0.;
	for j in range(N):
		deltatj = t - tj[j]
		if betaj[j]!=0:
			inv_betaj = 1./betaj[j];
			sol_nht += Aj[j]*np.power(inv_betaj,2.)*(np.exp(-deltatj/tauj[j])-np.exp(-gamma*deltatj) )*heaviside(deltatj); # For positive j
			sol_nht +=-Aj[j]*inv_betaj*np.exp(-gamma*deltatj)*deltatj*heaviside(deltatj)# For negative j
		else:
			sol_nht += 0.5*Aj[j]*(deltatj**2)*np.exp(-alpha*deltatj )*heaviside( deltatj )# Unique term j

	sol_nht += sigma_gas_0*np.exp(-gamma*t)*t*heaviside(t)# For "zero" j
	sol_nht = nuL*sol_nht; #Multiply by nuL

	sol_nht = yx*(1-R)*sol_nht; # Multiply by the terms in the diff. eq.
	# ---------End of the contribution of the trivial non-homogeneous term


	return( sol_nht )
# -------------------------# End of SolveChemEvolModel_InhomogeneousTrivialTerm -------------------------------------------------


# SolveChemEvolModel --------------------------------------------------------------------------------------------------
def SolveChemEvolModel(t, chemdict):
	'''
	"sigmaX_0"
	"omega"
	"yx"
	"R"
	"nuL"
	"tauj": Array/list
	"tj": Array/list
	"Aj":
	"sigma_gas_0":
	"tauD":
	"taup":
	"tauI":
	"tau0":
	"AG"
	"AE"
	"AI"
	"CIa":
	"sigma_p"
	"mx1a"
	"tau1"
	"tau2"
	'''
	# Check if the chemdict parameters are correct:
	chemdict = prepare_chemdict( chemdict );
	# Extract the parameters common for all the DTDs:
	sigmaX_0 = chemdict["sigmaX_0"]
	omega = chemdict["omega"];
	yx = chemdict["yx"];
	R = chemdict["R"];
	nuL = chemdict["nuL"];
	tauj = chemdict["tauj"]
	tj = chemdict["tj"];
	Aj = chemdict["Aj"]
	sigma_gas_0 = chemdict["sigma_gas_0"]
	#-------------------------------------
	# DTD constants
	mx1a = chemdict["mx1a"]
	CIa = chemdict["CIa"]
	#-------------------------------------
	# Gaussian DTD parameters
	AG_arr = chemdict.get("AG", [])	
	sigma_p_arr = chemdict["sigma_p"]
	taup_arr = chemdict["taup"]
	tau1G_arr = chemdict["tau1G"]
	tau2G_arr = chemdict["tau2G"]
	# Exponential DTD parameters
	AE_arr = chemdict.get("AE", [])
	tauD_arr = chemdict["tauD"]
	tau1E_arr = chemdict["tau1E"]
	tau2E_arr = chemdict["tau2E"]
	# Inverse DTD parameters
	AI_arr = chemdict.get("AI", [])
	tauI_arr = chemdict.get("tauI", len(AI_arr)*[1.]) # Since it is degenerated with Ai, we set it to one by default
	tau0_arr = chemdict["tau0"]
	tau1I_arr = chemdict["tau1I"]
	tau2I_arr = chemdict["tau2I"]
	# Get the number of Gaussian, Expo. and 1/t DTD terms
	Ng, Ne, Ni = len(AG_arr), len(AE_arr), len(AI_arr)
	#-------------------------------------
	# Safety checks:
	assert(len(tauj)==len(tj)), "tj, tauj lengths mismatch"
	assert(len(Aj)==len(tj)), "tj, Aj lengths mismatch"
	assert( (len(sigma_p_arr)==Ng) and  (len(taup_arr)==Ng) and (len(tau1G_arr)==Ng) and (len(tau2G_arr)==Ng)), "Dimensions of Gaussian DTD parameters mismatch"
	assert( (len(tauD_arr)==Ne) and  (len(tau1E_arr)==Ne) and (len(tau2E_arr)==Ne) ), "Dimensions of Exponential DTD parameters mismatch"
	assert( (len(tauI_arr)==Ni) and  (len(tau0_arr)==Ni) and  (len(tau1I_arr)==Ni) and (len(tau2I_arr)==Ni) ), "Dimensions of 1/t DTD parameters mismatch"
	#-------------------------------------

	# Useful definitions
	alpha = (1.+omega-R)*nuL
	# ----------------------------------------------------------------

	#			ANALYTIC SOLUTION

	# ---------------------------------------------------------------


	#print("Analytical solution...")
	# ------------------------------------------------------------------------
	# 1) Homogeneous solution term
	aux_homo = sigmaX_0*np.exp(-alpha*t)
	# ------------------------------------------------------------------------
	# 2) Non-Homogeneous trivial term:
	aux_nht = SolveChemEvolModel_InhomogeneousTrivialTerm(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, yx, R)
	# ------------------------------------------------------------------------
	# 3) Non-Homogeneous non-trivial gaussian term:
	aux_nhg = 0.; # Initial value
	for i in range(Ng):
		aux_nhg += SolveChemEvolModel_GaussianTerm(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, CIa, AG_arr[i], taup_arr[i], sigma_p_arr[i], tau1G_arr[i], tau2G_arr[i], mx1a)
	# ------------------------------------------------------------------------
	# 4) Non-Homogeneous non-trivial exponential term:
	aux_nhe = 0.; # Initial value
	for i in range(Ne):
		aux_nhe += SolveChemEvolModel_ExponentialTerm(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, CIa, AE_arr[i], tauD_arr[i], tau1E_arr[i], tau2E_arr[i], mx1a)
	# ------------------------------------------------------------------------
	# 5) Non-Homogeneous non-trivial inverse term:
	aux_nhi = 0.; # Initial value
	for i in range(Ni):
		aux_nhi += SolveChemEvolModel_InverseTerm(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, CIa, AI_arr[i], tauI_arr[i], tau0_arr[i], tau1I_arr[i], tau2I_arr[i], mx1a)

	# Global exact solution:
	sigmaX_exact = aux_homo + aux_nht + aux_nhg + aux_nhe + aux_nhi
	return( sigmaX_exact )
# -------------------------# End of SolveChemEvolModel -------------------------------------------------



# ChemicalSolutionVerifier --------------------------------------------------------------------------
def ChemicalSolutionVerifier(t, chemdict):
	'''Evaluates the difference between the numerical evaluation of the dsigma/dt term and the left-hand side term of the equation.
	ChemicalSolutionVerifier(t, chemdict)
	t: time (at least four values are required)'''
	assert(len(t)>3), "ERROR: More time nodes are needed"

	chemdict = prepare_chemdict( chemdict );
	# Sort the time array:
	t = np.sort(t);

	# Extract the values of the parameters:
	omega = chemdict["omega"];
	yx = chemdict["yx"];
	R = chemdict["R"];
	nuL = chemdict["nuL"];
	mx1a = chemdict["mx1a"]
	#-------------------------------------

	# Compute the SFR(t) and the TypeIa SNe rate
	psi_function = Get_psi(t, chemdict);
	R1a_function = R1a_analytic(t, chemdict);

	# Solve the Equation
	sigmaX = SolveChemEvolModel(t, chemdict);
	# Compute the right-hand side of the equation
	right_hand = -nuL*sigmaX*(1.+omega-R) + yx*(1.-R)*psi_function + mx1a*R1a_function;
	# Compute the num. derivative
	left_hand = np.gradient(sigmaX,t)

	# Mask the initial point because the numerical gradient is worse:
	left_hand[0] = np.nan;
	# Mask also the nodes at both sides of tj (ill-defined gradient)
	tj = chemdict["tj"];
	for k in range(len(tj)):
		if (tj[k]<t[0]) or (tj[k]>t[-1]): continue;#tj unused
		n = 0;
		while(n<(len(t)-1)):
			if (t[n]<=tj[k]) and (tj[k]<t[n+1]):
				left_hand[n:n+2] = np.nan
				break;
			n = n+1;
	# Mask the last point because the numerical gradient is worse:
	left_hand[-1] = np.nan;
	return(left_hand-right_hand);
#--------------------------ChemicalSolutionVerifier-----------------------------------





# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# 				 ORGANISATION FUNCTIONS
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# ---------------------------------------------------------------------------
def add_element(chemdict, element="Fe", sigmaX_0=0.):
	''' Adds the yields associated with the element "element".
	Also adds the given value of sigmaX_0.
	chemdict: chemical dictonary
	element: element name. For the moment, only O, Mg, Si and Fe.
	sigmaX_0: initial density of "element"'''
	element = element.lower();
	chemdict = chemdict.copy()

	assert(element in ["fe","o","si", "mg"]), "Element %s not included yet"%element

	chemdict["sigmaX_0"] = 0.; # Add sigmaX_0

	if element=="fe":
		# Specific values for the iron:
		chemdict["yx"] = 5.6E-4
		chemdict["mx1a"] = 6.26E-01
	if element =="o":
		# Specific values for the oxygen:
		chemdict["yx"] = 1.022E-2
		chemdict["mx1a"] = 1.43E-01
	if element=="si":
		# Specific values for the silicon:
		chemdict["yx"] = 8.5*1E-4
		chemdict["mx1a"] = 0.154
	if element=="mg":
		# Specific values for the magnesium:
		chemdict["yx"] = 0.0026 # Johnson & Weinberg (2020)
		chemdict["mx1a"] = 0.0 # Johnson & Weinberg (2020)
	return(chemdict);

# ------------------- End of add_element ---------------------------------


# ---------------------------------------------------------------------------
def _separate_cases(alpha, tauj, tj, Aj):
	''' Extract the cases in which tauj[j]==1/alpha.
	Output format: (tauj, tj, Aj, tj_alpha, Aj_alpha)
	The last two outputs correspond to the special case tauj[j]=1/alpha
	(tauj, tj, Aj, tj_alpha, Aj_alpha) = _separate_cases(alpha, tauj, tj, Aj)'''
	
	N = len(tauj);
	inv_alpha = 1./alpha; # The inverse of alpha

	# Now separate the case tauj==1./alpha:
	Aj_gorro = []
	tj_gorro = []
	new_Aj = []
	new_tj = []
	new_tauj = []
	for j in range(N):
		if tauj[j]==inv_alpha:
			Aj_gorro.append(Aj[j])
			tj_gorro.append(tj[j])
		else:
			new_Aj.append(Aj[j])
			new_tj.append(tj[j])
			new_tauj.append(tauj[j])
	return(new_tauj, new_tj, new_Aj, tj_gorro, Aj_gorro)
# ------------------- End of _separate_cases ---------------------------------



# prepare_chemdict ----------------------------------------------------------------------
def prepare_chemdict(chemdict):
	''' Creates arrays/list when necessary'''

	# Names of the key parameters - - - - - - - - - - - - - - - - - -
	fields_gauss = ["AG", "sigma_p", "taup", "tau1G", "tau2G"]
	fields_exp = ["AE", "tauD", "tau1E", "tau2E"]
	fields_inv = ["AI", "tauI", "tau0", "tau1I", "tau2I"]
	fields_infall = ["tj", "Aj", "tauj"]	
	fields = fields_gauss + fields_exp + fields_inv + fields_infall
	# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


	# Common tau1:
	if ("tau1" in chemdict.keys()) and not ("tau1G" in chemdict.keys()) and ("AG" in chemdict.keys()): chemdict["tau1G"] = chemdict["tau1"];
	if ("tau1" in chemdict.keys()) and not ("tau1E" in chemdict.keys()) and ("AE" in chemdict.keys()): chemdict["tau1E"] = chemdict["tau1"];
	if ("tau1" in chemdict.keys()) and not ("tau1I" in chemdict.keys()) and ("AI" in chemdict.keys()): chemdict["tau1I"] = chemdict["tau1"];

	# Common tau2:
	if ("tau2" in chemdict.keys()) and not ("tau2G" in chemdict.keys()) and ("AG" in chemdict.keys()): chemdict["tau2G"] = chemdict["tau2"];
	if ("tau2" in chemdict.keys()) and not ("tau2E" in chemdict.keys()) and ("AE" in chemdict.keys()): chemdict["tau2E"] = chemdict["tau2"];
	if ("tau2" in chemdict.keys()) and not ("tau2I" in chemdict.keys()) and ("AI" in chemdict.keys()): chemdict["tau2I"] = chemdict["tau2"];

	# Remove tau1, tau2:
	if ("tau1" in chemdict.keys()): chemdict.pop("tau1")
	if ("tau2" in chemdict.keys()): chemdict.pop("tau2")

	# Make all the inputs iterable  - - - - - - - - - - - - - - - - - -
	for k, name in enumerate( fields ):
		# Check the element is a list/iterable
		try:
			len(chemdict[name]);
		except:
			chemdict[name] = [chemdict.get(name,None)]# Fill with empty list not-included fields
			if chemdict[name][0] is None: chemdict[name] = []
	# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



	# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -




	# Check the parameters of the DTDs are included
	if "AG" in chemdict.keys():
		for k, name in enumerate( fields_gauss[1:] ): assert(name in chemdict.keys() ), "Error in Gaussian DTD: no " + name + " in input dictionary"

	if "AE" in chemdict.keys():
		for k, name in enumerate( fields_exp[1:] ): assert(name in chemdict.keys() ), "Error in Exp. DTD: no " + name + " in input dictionary"

	if "AI" in chemdict.keys():
		for k, name in enumerate( fields_inv[1:] ): assert(name in chemdict.keys() ), "Error in 1/t DTD: no " + name + " in input dictionary"

	if "Aj" in chemdict.keys():
		for k, name in enumerate( fields_infall[1:] ): assert(name in chemdict.keys() ), "Error in infall: no " + name + " in input dictionary"
	# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



	# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	for k, name in enumerate(fields_gauss[:-1]): assert(len(chemdict[name])==len(chemdict.get(fields_gauss[k+1]))), "Dimensions in gaussian DTD parameters mismatch"

	for k, name in enumerate(fields_exp[:-1]): assert(len(chemdict[name])==len(chemdict.get(fields_exp[k+1]))), "Dimensions in exponential DTD parameters mismatch"

	for k, name in enumerate(fields_inv[:-1]): assert(len(chemdict[name])==len(chemdict.get(fields_inv[k+1]))), "Dimensions in inverse DTD parameters mismatch"

	for k, name in enumerate(fields_infall[:-1]): assert(len(chemdict[name])==len(chemdict.get(fields_infall[k+1]))), "Dimensions in infall parameters mismatch"
	# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

	# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	# Sort the infall times
	order = np.argsort(chemdict["tj"])
	if len(order)!=0:
		for k, name in enumerate(fields_infall): chemdict[name] = list(np.array(chemdict[name])[order]);
	# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

	# Check tau1<=tau2:
	if "AG" in chemdict.keys():
		for k in range( len(chemdict["tau1G"]) ): assert(chemdict["tau1G"][k]<=chemdict["tau2G"][k]), "  ERROR: tau1G must be LOWER than tau2G"
	if "AE" in chemdict.keys():
		for k in range( len(chemdict["tau1E"]) ): assert(chemdict["tau1E"][k]<=chemdict["tau2E"][k]), "  ERROR: tau1E must be LOWER than tau2E"
	if "AI" in chemdict.keys():
		for k in range( len(chemdict["tau1I"]) ): assert(chemdict["tau1I"][k]<=chemdict["tau2I"][k]), "  ERROR: tau1I must be LOWER than tau2I"
	# Check tau0<tau1:
	if "AI" in chemdict.keys():
		for k in range( len(chemdict["tau1I"]) ): assert(chemdict["tau0"][k]<chemdict["tau1I"][k]), "  ERROR: tau0 must be LOWER than tau1I"

	return(chemdict)
# ------------------- End of prepare_chemdict ---------------------------------




# Helpful functions -----------------------------------------------------------


# ------------------------------------------
def QD(x, a, c):
	'''Used in A.7'''
	num = a*c*np.exp((c-a)/(c*a)*x)*(a!=c) -x*(a==c);
	den = (a-c)*(a!=c) +1.*(a==c);
	return(num/den);
# ------------------------------------------


# ------------------------------------------
def PD(x, a, b, c):
	'''Used in A.8'''
	num = a*b*c*np.exp((c-b)/(b*c)*x)*((a!=c) & (b!=c)) - a*x*(c==b) + b*((c-b)*x-c*b)*np.exp((c-b)/(b*c)*x)*(c==a);
	den = (a-c)*(b-c)*((a!=c) & (b!=c)) + (a-c)*(c==b) + (b-c)**2*(c==a)
	return(c*num/den);
# ------------------------------------------


# ------------------------------------------
def SD(x, a, c):
	'''Used in A.9'''
	num = (a**2*c**2)*np.exp( (c-a)/(a*c)*x )*(c!=a) + 0.5*x**2*(c==a)
	den = (a-c)**2*(c!=a) + 1.*(c==a)
	return(num/den);
# ------------------------------------------


# ------------------------------------------------
def QD_gorro(x,y,a,c):
	'''Aux. function associated with the exponential term of the DTD'''
	if(a!=c):
		result = a*c/(a-c)**2*(np.exp( (c-a)*x/c/a)*(c*a + (c-a)*(y-x)) )
	else:
		result = - 0.5*(y-x)**2
	return(result);
# ------------------------------------------------


# ------------------------------------------------
def SD_gorro(x,a,c):
	'''Aux. function associated with the exponential term of the DTD'''
	if(a!=c):
		aux = (c-a)/c/a
		result = (1./aux)**3*( -1. + np.exp( -aux*x )*(1. + aux*x + 0.5*(aux*x)**2) )
	else:
		result = -(x**3)/6.

	result = result*heaviside(x)
	return(result);
# -----------------------------------------------


# ------------------------------------------
def LatexCheckEqA9a(t, chemdict):
	''' Checks if R1a_gauss is correct in the Latex document:'''
	chemdict = prepare_chemdict( chemdict )

	# Extract the values of the parameters:
	omega = chemdict["omega"];
	R = chemdict["R"];
	nuL = chemdict["nuL"];
	tauj = chemdict["tauj"]
	tj = chemdict["tj"];
	N = len(tj)
	Aj = chemdict["Aj"]
	sigma_gas_0 = chemdict["sigma_gas_0"]
	# DTD
	CIa = chemdict["CIa"]
	#-------------------------------------


	#-------------------------------------
	# Safety checks:
	assert(len(tauj)==len(tj)), "tj, tauj lengths mismatch"
	assert(len(Aj)==len(tj)), "tj, Aj lengths mismatch"
	#-------------------------------------

	# Useful definitions
	alpha = (1.+omega-R)*nuL
	inv_alpha = 1./alpha;
	

	# 1) R1a Gaussian term
	# ---------
	# Gaussian DTD parameters
	AG_arr = chemdict["AG"]
	Ng = len(AG_arr)# Get the number of Gaussians
	sigma_p_arr = chemdict["sigma_p"]
	taup_arr = chemdict["taup"]
	tau1G_arr = chemdict["tau1G"]
	tau2G_arr = chemdict["tau2G"]

	assert( (len(sigma_p_arr)==Ng) and  (len(taup_arr)==Ng) and (len(tau1G_arr)==Ng) and (len(tau2G_arr)==Ng)), "Dimensions of Gaussian DTD parameters mismatch"

	R1a_g = 0. # Cumulative value
	for i in range(Ng):
		AG = AG_arr[i]
		taup = taup_arr[i]
		sigma_p = sigma_p_arr[i]
		tau1 = tau1G_arr[i]
		tau2 = tau2G_arr[i]

		R1a_g += R1a_analytic_gaussian(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, CIa, AG, taup, sigma_p, tau1, tau2)


	R1a_g_latex = 0.;
	for i in range(Ng):
		AG = AG_arr[i]
		taup = taup_arr[i]
		sigma_p = sigma_p_arr[i]
		tau1 = tau1G_arr[i]
		tau2 = tau2G_arr[i]
		for j in range(N):
			deltatj = t-tj[j]
			Kj = Aj[j]*nuL/(alpha-1./tauj[j])
			mint1 = tau1*(tau1<deltatj) + deltatj*(deltatj<=tau1)
			mint2 = tau2*(tau2<deltatj) + deltatj*(deltatj<=tau2)
			etaj = taup + sigma_p**2/tauj[j]
			etaalpha = taup + sigma_p**2*alpha
			R1a_g_latex += CIa*AG*sigma_p*np.sqrt(0.5*np.pi)*Kj*np.exp(-(deltatj-taup-0.5*sigma_p**2/tauj[j])/tauj[j])*( erf((mint2-etaj)/(np.sqrt(2.)*sigma_p))-erf((mint1-etaj)/(np.sqrt(2.)*sigma_p)) )
			R1a_g_latex -= CIa*AG*sigma_p*np.sqrt(0.5*np.pi)*Kj*np.exp(-alpha*(deltatj-taup-sigma_p**2*alpha*0.5) )*( erf((mint2-etaalpha)/(np.sqrt(2.)*sigma_p))-erf((mint1-etaalpha)/(np.sqrt(2.)*sigma_p)) )
		mint1 = tau1*(tau1<t) + t*(t<=tau1)
		mint2 = tau2*(tau2<t) + t*(t<=tau2)
		R1a_g_latex += CIa*AG*sigma_p*np.sqrt(0.5*np.pi)*sigma_gas_0*nuL*np.exp(-alpha*(t-taup-sigma_p**2*alpha*0.5))*( erf((mint2-etaalpha)/(np.sqrt(2.)*sigma_p))-erf((mint1-etaalpha)/(np.sqrt(2.)*sigma_p)) )
	# Prepare the output
	return(R1a_g,R1a_g_latex)
# -------------------------# End of LatexCheckEqA9a -------------------------------------------------


# ------------------------------------------
def LatexCheckEqA9b(t, chemdict):
	''' Checks if R1a_exp is correct in the Latex document:'''
	chemdict = prepare_chemdict( chemdict )

	# Extract the values of the parameters:
	omega = chemdict["omega"];
	R = chemdict["R"];
	nuL = chemdict["nuL"];
	tauj = chemdict["tauj"]
	tj = chemdict["tj"];
	N = len(tj)
	Aj = chemdict["Aj"]
	sigma_gas_0 = chemdict["sigma_gas_0"]
	# DTD
	CIa = chemdict["CIa"]
	#-------------------------------------


	#-------------------------------------
	# Safety checks:
	assert(len(tauj)==len(tj)), "tj, tauj lengths mismatch"
	assert(len(Aj)==len(tj)), "tj, Aj lengths mismatch"
	#-------------------------------------

	# Useful definitions
	alpha = (1.+omega-R)*nuL
	inv_alpha = 1./alpha;
	

	# 1) R1a Exponential term
	# ---------
	# Exponential DTD parameters
	AE_arr = chemdict["AE"]
	Ne = len(AE_arr)# Get the number of Exponentials
	tauD_arr = chemdict["tauD"]
	tau1E_arr = chemdict["tau1E"]
	tau2E_arr = chemdict["tau2E"]

	assert( (len(tauD_arr)==Ne) and  (len(tau1E_arr)==Ne) and (len(tau2E_arr)==Ne) ), "Dimensions of Exponential DTD parameters mismatch"

	R1a_e = 0.# Cumulative value
	for i in range(Ne):
		AE = AE_arr[i]
		tauD = tauD_arr[i]
		tau1 = tau1E_arr[i]
		tau2 = tau2E_arr[i]

		R1a_e += R1a_analytic_exponential(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, CIa, AE, tauD, tau1, tau2);


	R1a_e_latex = 0.;
	for i in range(Ne):
		AE = AE_arr[i]
		tauD = tauD_arr[i]
		tau1 = tau1E_arr[i]
		tau2 = tau2E_arr[i]
		for j in range(N):
			deltatj = t-tj[j]
			Kj = Aj[j]*nuL/(alpha-1./tauj[j])
			mint1 = tau1*(tau1<deltatj) + deltatj*(deltatj<=tau1)
			mint2 = tau2*(tau2<deltatj) + deltatj*(deltatj<=tau2)
			R1a_e_latex += CIa*AE*Kj*np.exp(-deltatj/tauj[j])*(QD(mint1, tauj[j], tauD)-QD(mint2, tauj[j], tauD))
			R1a_e_latex -= CIa*AE*Kj*np.exp(-alpha*deltatj)*(QD(mint1, inv_alpha, tauD)-QD(mint2, inv_alpha, tauD))
		mint1 = tau1*(tau1<t) + t*(t<=tau1)
		mint2 = tau2*(tau2<t) + t*(t<=tau2)
		R1a_e_latex += CIa*AE*sigma_gas_0*nuL*np.exp(-alpha*t)*(QD(mint1, inv_alpha, tauD)-QD(mint2, inv_alpha, tauD))
	# Prepare the output
	return(R1a_e,R1a_e_latex)
# -------------------------# End of LatexCheckEqA9b -------------------------------------------------



# ------------------------------------------
def LatexCheckEqA9c(t, chemdict):
	''' Checks if R1a_inv is correct in the Latex document:'''
	chemdict = prepare_chemdict( chemdict )

	# Extract the values of the parameters:
	omega = chemdict["omega"];
	R = chemdict["R"];
	nuL = chemdict["nuL"];
	tauj = chemdict["tauj"]
	tj = chemdict["tj"];
	N = len(tj)
	Aj = chemdict["Aj"]
	sigma_gas_0 = chemdict["sigma_gas_0"]
	# DTD
	CIa = chemdict["CIa"]
	#-------------------------------------


	#-------------------------------------
	# Safety checks:
	assert(len(tauj)==len(tj)), "tj, tauj lengths mismatch"
	assert(len(Aj)==len(tj)), "tj, Aj lengths mismatch"
	#-------------------------------------

	# Useful definitions
	alpha = (1.+omega-R)*nuL
	inv_alpha = 1./alpha;
	

	# 1) R1a Inverse term
	# ---------
	# Inverse DTD parameters
	AI_arr = chemdict["AI"]
	Ni = len(AI_arr)# Get the number of Inverses
	tauI_arr = chemdict["tauI"]
	tau0_arr = chemdict["tau0"]
	tau1I_arr = chemdict["tau1I"]
	tau2I_arr = chemdict["tau2I"]

	assert( (len(tauI_arr)==Ni) and  (len(tau0_arr)==Ni) and  (len(tau1I_arr)==Ni) and (len(tau2I_arr)==Ni) ), "Dimensions of 1/t DTD parameters mismatch"

	R1a_i = 0.# Cumulative value
	for i in range(Ni):
		AI = AI_arr[i]
		tauI = tauI_arr[i]
		tau0 = tau0_arr[i]
		tau1 = tau1I_arr[i]
		tau2 = tau2I_arr[i]

		R1a_i += R1a_analytic_inverse(t, alpha, nuL, Aj, tauj, tj, sigma_gas_0, CIa, AI, tauI, tau0, tau1, tau2)


	R1a_i_latex = 0.;
	for i in range(Ni):
		AI = AI_arr[i]
		tauI = tauI_arr[i]
		tau0 = tau0_arr[i]# The shift cannot be applied before because R1a_analytic_inverse use the original time, tau1 and tau2
		tau1 = tau1I_arr[i]
		tau2 = tau2I_arr[i]
		for j in range(N):
			deltatj = t-tj[j]# Shift in deltatj applied here
			Kj = Aj[j]*nuL/(alpha-1./tauj[j])
			mint1 = tau1*(tau1<deltatj) + deltatj*(deltatj<=tau1)
			mint2 = tau2*(tau2<deltatj) + deltatj*(deltatj<=tau2)
			R1a_i_latex += CIa*AI*tauI*Kj*np.exp(-(deltatj-tau0)/tauj[j])*( _safeexpi((mint2-tau0)/tauj[j])-_safeexpi((mint1-tau0)/tauj[j]) )
			R1a_i_latex += CIa*AI*tauI*Kj*np.exp(-alpha*(deltatj-tau0))*( _safeexpi(alpha*(mint1-tau0))-_safeexpi(alpha*(mint2-tau0)) )
		mint1 = tau1*(tau1<t) + t*(t<=tau1)
		mint2 = tau2*(tau2<t) + t*(t<=tau2)
		R1a_i_latex += CIa*AI*tauI*sigma_gas_0*nuL*np.exp(-alpha*(t-tau0))*(_safeexpi(alpha*(mint2-tau0))-_safeexpi(alpha*(mint1-tau0)))# Shift in t applied here
	# Prepare the output
	return(R1a_i,R1a_i_latex)
# -------------------------# End of LatexCheckEqA9c -------------------------------------------------




# --------------------------------------------------------------------
def Load_MR01_dict(dict_MR01=None, gamma=0.5):
	''' Automatises the loading of the FIT of the MR01 DTD. Adds the DTD fields to the dict_MR01 dictionary (if it does not exist, create it).
	gamma only 2. or 0.5'''

	if (gamma==0.5):
		dict_part1 = {'AG': [-0.9507374, 0.18962885, -0.05963268], 'AE': [20.908981,-76.84657,61.405315,65.75655,-67.931114,0.032177716,-1.052533,1.7726375,-1.1302925], 'AI': [-0.058720887, 0.8307219], 'taup': [0.093549225, 0.23066255759579082, 1.2836338207977886], 'sigma_p': [0.20250127381009236, 0.12941138077718786, 0.1955465805271096],'tauD':[0.575242740268056,1.150485480536112,1.725728220804168,2.300970961072224,2.87621370134028,1.7949875142962848,3.5899750285925696,5.384962542888855,7.179950057185139],
 'tauI': 2*[1.0], 'tau1G': 3*[0.03], 'tau2G': 3*[1.612], 'tau1E': 5*[0.03]+4*[1.612], 'tau2E': 5*[1.612]+4*[13.794999999999998], 'tau1I': [0.03, 1.612], 'tau2I': [1.612, 13.794999999999998], 'tau0': 2*[0.0]}
	if (gamma==2.):
		dict_part1 = {'AG': [-1.0942523, -0.17504661, 0.013960359, 0.076081686, -0.064493224], 'AE': [25.947952,  -109.78866,  173.25157,  -87.19557,  1.7665788,  -1.1984636,  2.4828148,  -7.8752403,  10.152289,  -4.8747625], 'AI': [-0.06924627, 1.3518673], 'taup': [0.07875, 0.34, 0.395, 0.78, 1.355], 'sigma_p': [0.13203429428173546,  0.07771071405060358,  0.11511004421963635,  0.11369845187749276,  0.1258690453607682], 'tauD': [0.5047115446616498,  1.0094230893232996,  1.5141346339849493,  2.018846178646599,  2.523557723308249,  1.443839515065346,  2.887679030130692,  4.3315185451960385,  5.775358060261384,  7.21919757532673], 'tauI': 2*[1.0], 'tau1G': 5*[0.03], 'tau2G': 5*[1.612], 'tau1E': 5*[0.03] + 5*[1.612], 'tau2E': 5*[1.612] + 5*[13.794999999999998], 'tau1I': [0.03, 1.612], 'tau2I': [1.612, 13.794999999999998], 'tau0': 2*[0.0]}

	if dict_MR01 is None: dict_MR01 = dict(); # Create the dictionary
	dict_MR01 = dict_MR01.copy(); # Create a copy but free the pointer

	# Merge both dictionaries:
	for key in dict_part1.keys(): dict_MR01[key] = list(dict_part1[key])

	# Remove the separated components
	del(dict_part1)

	return(dict_MR01)
# ------------------------ Load_MR01_dict ----------------------



# --------------------------------------------------------------------
def Load_G05Wide_dict(dict_G05=None):
	''' Returns the dictionary with the fit of the Wide Greggio05 DTD.'''

	# Fit of the G05 Wide DTD
	#------------------------------------------------------------
	dict_part1 = {'AG': [-0.011604926,-0.063981086,0.07694069,0.017263718,-0.09277936, 0.11850091], 'AE': [3.4356658, -2.018978,  0.27029914,-0.65370893,0.64225656,1.5200671,-3.3111565,1.6912478],'AI': [-0.060610272, 0.03796646],'taup': [0.093549225, 0.40217375000000005,
  0.40217375000000005,0.40217375000000005,0.40217375000000005,0.40217375000000005],'sigma_p': [0.09438804914646945, 0.1, 0.2,0.30000000000000004, 0.4, 0.5],'tauD': [3.18751243084999, 6.37502486169998, 2.8043023208094295, 5.360826275431208, 7.91386350732254,10.465975747057124,13.017708749840494,15.56924947703517],'tau0': [0.0, 0.35],'tauI': [1.0, 1.0],'tau1G': [0.044] + 5*[0.40000000000000000],'tau2G': [0.4, 13.8, 13.8, 13.8, 13.8, 13.8],'tau1E': [0.044,  0.044] + 6*[ 0.40000000000000000],'tau2E': [0.4, 0.4, 13.8, 13.8, 13.8, 13.8, 13.8, 13.8],'tau1I': [0.044, 0.40000000000000000],'tau2I': [0.4, 13.8]}

	if dict_G05 is None: dict_G05 = dict(); # Create the dictionary
	dict_G05 = dict_G05.copy(); # Create a copy but free the pointer

	# Merge both dictionaries:
	for key in dict_part1.keys(): dict_G05[key] = list(dict_part1[key])

	# Remove the separated components
	del(dict_part1)

	return(dict_G05)
# ------------------------ Load_G05Wide_dict ----------------------




# --------------------------------------------------------------------
def Load_G05Close_dict(dict_G05=None):
	''' Returns the dictionary with the fit of the Close Greggio05 DTD.'''

	# Fit of the G05 Close DTD
	#------------------------------------------------------------
	dict_part1 = {'AG': [-0.01841404, 0.06908514, 0.07991288, 0.020217998, 0.041908335, 0.031064995, 1.4102234e-01,  2.2727300e-03, -2.9716764e-02, -1.6765862e-03,-3.0695251e-03, -9.5378557e-05], 'AE': [9.916966, -0.2288622, -34.91759, 29.04085, 0.22326148, 1.7126161, -0.04215201, 0.030501552, 0.10006145], 'AI': [-0.12746474, 0.5041903, -1.176218, 0.68416184], 'taup': [0.102, 0.24376942021415324, 0.3031802465283689, 0.18479625086078738, 0.36450436801126646, 0.08144210206929202, 0.398, 3.003, 1.323, 6.003, 4.603, 7.803], 'sigma_p': [0.02086734262263574, 0.023009267549061577, 0.023646549558257138, 0.022027233578999324, 0.02031287893282801, 0.01875422613319202, 0.1, 1.316, 1.806, 2.0653, 1.473, 1.873], 'tauD': [0.27143054928643034, 0.5428610985728607, 0.814291647859291, 1.0857221971457214, 0.14206351597250458, 0.2445748152084426, 0.5212460031729189, 0.9273293242212868, 2.174904211184498], 'tau0': [0.0, 0.33, 0.3, 0.25], 'tauI': [1.0, 1.0, 1.0, 1.0], 'tau1G': [0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.398, 0.398, 0.398, 0.398, 0.398, 0.398], 'tau2G': [0.398, 0.398, 0.398, 0.398, 0.398, 0.398, 13.8, 13.8, 13.8, 13.8, 13.8, 13.8], 'tau1E': [0.042, 0.042, 0.042, 0.042, 0.398, 0.398, 0.398, 0.398, 0.398], 'tau2E': [0.398, 0.398, 0.398, 0.398, 13.8, 13.8, 13.8, 13.8, 13.8], 'tau1I': [0.042, 0.398, 0.398, 0.398], 'tau2I': [0.398, 13.8, 13.8, 13.8]}

	if dict_G05 is None: dict_G05 = dict(); # Create the dictionary
	dict_G05 = dict_G05.copy(); # Create a copy but free the pointer

	# Merge both dictionaries:
	for key in dict_part1.keys(): dict_G05[key] = list(dict_part1[key])

	# Remove the separated components
	del(dict_part1)

	return(dict_G05)
# ------------------------ Load_G05Close_dict ----------------------




# --------------------------------------------------------------------
def Load_P08_dict(dict_P08=None):
	''' Returns the dictionary with the Pritchet08 DTD'''

	dict_part1 = {'AG': [-0.1534861], 'AE': [ 0.6854898, -3.376085 ,  5.5749445, -2.7450755], 'AI': [-0.02623133,  0.31474027,-0.8276519 ,  0.599177  ], 'taup': [0.0035], 'sigma_p': [0.1], 'tauD': [5.564998502546989,  11.074999247565016,  16.584999497528703,
  22.094999622828002], 'tau0': [0.025, 0.02, 0.015, 0.01], 'tauI': [1.0, 1.0, 1.0, 1.0], 'tau1G': [0.03], 'tau2G': [13.8], 'tau1E': [0.03, 0.03, 0.03, 0.03], 'tau2E': [13.8, 13.8, 13.8, 13.8], 'tau1I': [0.03, 0.03, 0.03, 0.03], 'tau2I': [13.8, 13.8, 13.8, 13.8]}

	if dict_P08 is None: dict_P08 = dict(); # Create the dictionary
	dict_P08 = dict_P08.copy(); # Create a copy but free the pointer

	# Merge both dictionaries:
	for key in dict_part1.keys(): dict_P08[key] = list(dict_part1[key])

	# Remove the separated components
	del(dict_part1)

	return(dict_P08)
# ------------------------ Load_P08_dict ----------------------



# --------------------------------------------------------------------
def Load_T08_dict(dict_T08=None):
	''' Returns the dictionary with the Totani08 DTD'''

	dict_part1 = {'AI': [1.  ], 'tau0': [.0], 'tauI': [1.0], 'tau1I': [0.1], 'tau2I': [10.]}

	if dict_T08 is None: dict_T08 = dict(); # Create the dictionary
	dict_T08 = dict_T08.copy(); # Create a copy but free the pointer

	# Merge both dictionaries:
	for key in dict_part1.keys(): dict_T08[key] = list(dict_part1[key])

	# Remove the separated components
	del(dict_part1)

	return(dict_T08)
# ------------------------ Load_T08_dict ----------------------



# --------------------------------------------------------------------
def Load_S05_dict(dict_S05=None):
	''' Returns the dictionary with the Strolger05 DTD'''
	dict_part1 = {'AG': [1.], 'taup': [3.4], 'sigma_p': [0.68], 'tau1G': [0.25], 'tau2G': [13.8]}

	if dict_S05 is None: dict_S05 = dict(); # Create the dictionary
	dict_S05 = dict_S05.copy(); # Create a copy but free the pointer

	# Merge both dictionaries:
	for key in dict_part1.keys(): dict_S05[key] = list(dict_part1[key])

	# Remove the separated components
	del(dict_part1)

	return(dict_S05)
# ------------------------ Load_S05_dict ----------------------




# --------------------------------------------------------------------
def Load_MVP06_dict(dict_MVP06=None):
	''' Returns the dictionary with the Mannucci+06 DTD'''
	dict_part1 = {'AG': [19.95], 'AE': [0.17], 'taup': [0.05], 'sigma_p': [0.01], 'tauD': [3.], 'tau1G': [0.03], 'tau2G': [10.05], 'tau1E': [0.03], 'tau2E': [10.05]}

	if dict_MVP06 is None: dict_MVP06 = dict(); # Create the dictionary
	dict_MVP06 = dict_MVP06.copy(); # Create a copy but free the pointer

	# Merge both dictionaries:
	for key in dict_part1.keys(): dict_MVP06[key] = list(dict_part1[key])

	# Remove the separated components
	del(dict_part1)

	return(dict_MVP06)
# ------------------------ Load_MVP06_dict ----------------------
