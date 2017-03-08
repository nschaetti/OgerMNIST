#include <math.h>
#include "mex.h"

double f_nl(double x, double *p)

{
  int k, type_NL;
  double beta, phi, alpha, a ,fc , y=0, yi=0;

  type_NL = p[0];	/* 0: sinus, 1: Fabry-Pérot, 2: RLCs, 3: polynôme */
  beta = p[1];     /* poids ou amplification de la NL */
  phi = p[2];     /* décalage le long des abscisses */
  alpha = p[3];	 /* contraste, finesse, ou fi, Qi et Gi */


  switch (type_NL) {
	case 1:			 /*lame Fabry-Pérot*/
  		y = sin(x + phi);
   	y = beta / (1.0 + alpha * y * y);
	break;

	case 2:			 /*pics de résonance RLC*/
		for (k = 0; k < alpha; k++) {
		  	a = (phi + x) / *(p+4+3*k);
			a -= 1/a;
			fc = *(p+6+3*k) / (1 / (*(p+5+3*k) * *(p+5+3*k)) + a * a);
	      y += a * fc;
	      yi -= fc / *(p+5+3*k);
	   }
	   y = beta * sqrt(y * y + yi * yi);
	break;

	case 3:			 /*polynôme "borné"*/
      a = x * x;
		y = -beta * x * (1.0 + 2.0*x + 8.0*a) / (1.0 + 16.0*a*a);
	break;

  	default :			 /*cristal biréfringent*/
	     	y = 0.5 * beta * (1 - alpha * cos(2.0 * (x + phi)));
	break;
  }
  return y;
}

/******************************************************************************/

/* Intégration numérique pas à pas par Runge-Kutta d'ordre 4.
 *
 * L'unité de temps est le pas d'intégration h=1.
 * Le retard T est supposé être un multiple entier du pas d'intégration (!?).
 *
 * les entrées-sorties sont:
 *		- un pointeur *x sortant la s�rie temporelle résultat du calcul
 *    - un pointeur *ci entrant les conditions initiales, taille nT+1
 *    - un pointeur *tps entrant les paramètres temporels ([0]: constante
 *		de temps tau, [1]: retard temporel T, [2]: durée de la phase transitoire,
 *    [3]: durée d'intégration après le transitoire, donc longueur du
 *    vecteur de sortie pointé par x). L'unité de base de ces paramètres est
 *		le pas d'échantillonnage, ou itération du calcul pas à pas.
 *    - un pointeur *param définissant la fonction non linéaire (cf.
 *		la fonction f_nl plus haut). *(param+4) est la boucle ouverte ou
 *      fermée selon la valeur 0 ou 1 respectivement.
 *    - le pointeur *b contient une excitation extérieure de la dynamique
 *      à retard (soit du bruit, soit une information forçant la dynamique
 *      comme dans les applications NTC. Cette excitation a une durée
 *      équivalente à la durée de calcul de la dynamique.
 *
 */

void int_rk4(double *y, double *ci, double *tps, double *param, double *b)
{
  int i, j, m, ntint, nttrans, nT;
  double tau, dy0, dy05, dy05s, dy1, y05, y1, fpy;


  tau  	 = *tps;       				   /* constante de temps */
  nT      = (int) *(tps+1);			  /* retard T */
  nttrans = (int) *(tps+2) * nT;	 /* durée phase transitoire */
  ntint   = (int) *(tps+3) * nT; 	/* durée de calcul après transitoire */

  m = 0;
  j = 0;

  while (j<(nttrans + ntint - 1)) {

      if (j==0)              /* si première durée T à calculer...*/
      	*y = *(ci+nT);		/*...affecte l'adresse correcte au pointeur */

      else if (j<(nttrans+1)) {      /* si transitoire...*/
         *ci = *(ci+nT);            /*...mise à jour des...*/
         *y = *(y+nT);

         for (i=1; i<(nT+1); i++)       /*...conditions initiales */
         	*(ci+i) = *(y+i);
      }
      else {				 		/* si après transitoire...*/
      	m = j - nttrans;           /*...fixe l'indice courant...*/
         ci = &(*(y+m-nT));       /*...des échantillons à enregistrer...*/
      }							 /*...met à jour le pointeur des ci*/

		/* calcul sur une durée T */
      if (*(param+4)==1)
        fpy = f_nl(*ci + *(b+m), param); /* ajout du bruit */
      else
        fpy = f_nl(*(b+m), param); /* ajout du bruit seul, sans feedback */
      
      for (i=0; i<nT; i++) {

         dy0 = - *(y+m+i) +  fpy;
         y05 = *(y+m+i) + 0.5 / tau * dy0;
         
         if (*(param+4)==1)
            fpy = f_nl(0.5 * (*(ci+i+1) + *(ci+i) + *(b+m+i) + *(b+m+i+1)), param); /* ajout du bruit */
         else
             fpy = f_nl(0.5 * (*(b+m+i) + *(b+m+i+1)), param); /* ajout du bruit, sans feedback */
         
         dy05 = -y05 + fpy;
         y05 = *(y+m+i) + 0.5 / tau * dy05;
         dy05s = -y05 + fpy;
         y1 = *(y+m+i) + dy05s / tau;
         
         if (*(param+4)==1)
            fpy = f_nl(*(ci+i+1) + *(b+m+i+1), param) ; /* ajout du bruit */
         else
            fpy = f_nl(*(b+m+i+1), param) ; /* ajout du bruit seul, sans feedback */
         
         dy1 = -y1 + fpy;

         *(y+m+i+1) = *(y+m+i) + (dy0 + dy1 + 2.0 * (dy05 + dy05s)) / 6.0 / tau;

	      j++;
      }      /* fin d'une durée T */
  }
  /* fin de l'intégration */

}

/******************************************************************************/


/*---------*/
/* gateway */
/*---------*/
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
  double *y ,*ci, *tps, *param, *b;
  int     i, m;

  /*  Teste le nombre correct d'arguments */
  if(nrhs != 4)
    mexErrMsgTxt("Il faut 4 entrées.");
  if(nlhs != 1)
    mexErrMsgTxt("Une sortie seulement.");

  /* Teste la nature des entrées */
  for (i = 0; i < nrhs; i++)
  	{
     	if( !mxIsNumeric(prhs[i]) || !mxIsDouble(prhs[i]) ||
      mxIsEmpty(prhs[i]) || mxIsComplex(prhs[i]) ||
 	   mxGetN(prhs[i])!=1 )
        {
 		   	mexErrMsgTxt("Vecteurs d'entrée incorrects.");
        }
   }

  /*  Récupère les vecteurs d'entrée et leur taille  */
  ci = mxGetPr(prhs[0]);
  m = mxGetM(prhs[0]);
  tps = mxGetPr(prhs[1]);
  param = mxGetPr(prhs[2]);
  b = mxGetPr(prhs[3]);

  if (*(tps+1) != (m - 1)) /* il faut (nT+1) conditions initiales */
  	{
  		mexErrMsgTxt("Taille des conditions initiales incorrecte.");
   }

  if ( mxGetM(prhs[1])!=4 )
  	{
  		mexErrMsgTxt("Il faut 4 paramètres temporels.");
   }

  if (mxGetM(prhs[2]) < 5)
  	{
  		mexErrMsgTxt("5 paramètres minimum pour la fonction non linéaire.");
   }

  /*  Fixe la dimension du vecteur de sortie */
  plhs[0] = mxCreateDoubleMatrix((m - 1) * *(tps+3) + 1, 1, mxREAL);


  /*  Crée le vecteur de sortie */
  y = mxGetPr(plhs[0]);

  /* Appelle la fonction de calcul en C */
  int_rk4(y,ci,tps,param,b);
}
