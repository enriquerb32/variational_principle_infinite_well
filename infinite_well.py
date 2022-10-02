"""
/***********************************************************************************************************************/                                                                            					   			 							         		
/* Purpose: The infite well is one of the most used potential functions in order to model 
            simple interactions between particles, such as electrons in atoms. Here, knowing
            beforehand the analytical solution for the Schrödinger’s equation, we apply
            the Variational Principle to calculate numerical solutions for this potential.
            This principle when a fitting test funcion is chosen constitutes a powerful tool
            regarding fields such as quantum physics and Physical-Chemistry. In this work
            the ground state and first excited states will be calculated and their energies
            compared with those obtained with the analytical solution.
/***********************************************************************************************************************/
"""

import numpy as np
from matplotlib import pyplot as plt
import sympy as sy
from scipy.optimize import minimize
from sympy.utilities.lambdify import lambdify

# Variational principle
# Loop for the whole program
Nc = np.arange ( 1 , 6 , 1 ) # Every N from 1 to 5
nc = np.arange ( 1 , 4 , 1 ) # Every n from 1 to 3
Nrep = [ ]
for z in range ( len (Nc) ) :
    Nrep.append(Nc[z])
cont = 0
for s in range ( len ( nc ) ) :
    EN = [ ] #Contains the ground state energies for each N
    for v in range ( len (Nc ) ) :
        # Parameters
        a = 1 #well width
        hb = 6.62607015*10**(-34)/(2*np.pi)
        m = 1 #mass
        n = nc [ s ] # energy level in the well
        N = Nc [ v ] # coefficient on test function
        print ( n , N)
        an1= np.ones ( [N+1 ] ) # seed minimization vector
        an = [ ] # Contains coefficients
        x = sy.Symbol ( 'x' ) # symbolic x
        
        # We define los coefficients a_i

        for i in range (N + 1 ) :
            t = i
            a1 = sy.Symbol ('a%d'%(t))
            an . append ( a1 )
            
        # Functions

        def fpozoana ( n , t ) : #Analytical function of infinite well
            return -np.sqrt ( 2 / a )*np.sin (n*np.pi * ( t-a / 2 ) / ( a ) )
        def Epozoana ( n ) : # Analytical energy of the well
            return ( n**2*np.pi**2*hb**2 )/( 2*m*a )
        def fprueba ( x , n ) : # Test function
            if n == 1 :
                f = [ ]
                for i in range (N+ 1 ) :
                    j = an [ i ]*( ( a /2-x )**(N-i +1 )*( x+a / 2 )**( i +1 ) )
                    f.append ( j )
                return f
            elif n == 2 :
                f = [ ]
                for i in range (N+ 1 ) :
                    j = an [ i ]*-x * ( ( a/2 - x )**(N-i +1 )*( x+a / 2 )**( i +1 ) )
                    f.append ( j )
                return f
            elif n == 3 :
                f = [ ]
                for i in range (N+ 1 ) :
                    j = an [ i ]*-( x**2 - 1/35)*( ( a /2-x )**(N-i +1 ) * ( x+a / 2 ) ** ( i +1 ) )
                    f.append ( j )
                return f

        k = fprueba ( x , n ) # The function is summed in the form of a list
        u = 0
        for i in range ( len ( k ) ) :
            u += k [ i ]
            
        derfp = sy.diff ( u , x , 2 ) # Second derivative in function of x

        u2 = u*u # Squared test function

        f = derfp*u # Twice derivated function * function
        
        enum = sy.integrate ( f , ( x,-a / 2 , a / 2 ) ) # energy numerator
        ediv = sy.integrate ( u2 , ( x,-a / 2 , a / 2 ) ) # energy denominator
        e = -hb**2 / ( 2*m)*enum / ediv # energy
        
# We need to parametrize the energy in order to use it through minimize
# an [ : ] iterates over the array of coefficients ai and assignates them to e
# This way, it is possible to operate with floats and minimize

        ep = lambdify ( ( an [ : ] ) , e )
        
# This function assignates variables xi to each associated parameter 
# The way to do it is as a vector with a component por each ai 
# an [ : ] will act as a seed in minimize for each of these vectors

        def epv ( x ) :

            if N == 1 :
                return ep ( x [ 0 ] , x [ 1 ] )
            elif N == 2 :
                return ep ( x [ 0 ] , x [ 1 ] , x [ 2 ] )
            elif N == 3 :
                return ep ( x [ 0 ] , x [ 1 ] , x [ 2 ] , x [ 3 ] )
            elif N == 4 :
                return ep ( x [ 0 ] , x [ 1 ] , x [ 2 ] , x [ 3 ] , x [ 4 ] )
            elif N == 5 :
                return ep ( x [ 0 ] , x [ 1 ] , x [ 2 ] , x [ 3 ] , x [ 4 ] , x [ 5 ] )
        
        # We have also tried the Powell minimizing method, but it performs a worse fit
        
        res = minimize ( epv , an1 , method ="Nelder-mead" , tol =1e-8).x

        def fprufin ( t ) : # test function with minimized coefficients
            tot = 0
            if n == 1:
                for i in range (N+ 1 ) :
                    k = res [ i ] * ( ( a /2 - t ) ** (N-i +1 ) * ( t +a / 2 ) ** ( i +1 ) )
                    tot += k
                return tot
            elif n == 2 :
                for i in range (N+ 1 ) :
                    k = res [ i ]*t * ( ( a /2 - t ) ** (N-i +1 ) * ( t +a / 2 ) ** ( i +1 ) )
                    tot += k
                return tot
            elif n == 3 :
                for i in range (N+ 1 ) :
                    k = res [ i ]*( t**2 - 1/35)*( ( a /2 - t ) ** (N-i +1 ) * ( t +a / 2 ) ** ( i +1 ) )
                    tot += k
                return tot
            
        t = np.arange (-a / 2 , a / 2 , 0.01 ) # coordinates vector
        h = [ ] # it will contain the tuple of coefficients and value for each of them ( ai , valor )
        # We obtain the squared minimized function through these tuples

        for i in range ( len ( res ) ) :
            h.append ( ( an [ i ] , res [ i ] ) )
        for i in range ( len ( h ) ) :
            if i == 0 :
                u2n = u2
                ei = e
            u2p = u2n.subs ( h [ i ] [ 0 ] , h [ i ] [ 1 ] )
            et = ei.subs ( h [ i ] [ 0 ] , h [ i ] [ 1 ] )
            ei=et
            u2n = u2p
            
        norm = sy.integrate ( u2n , ( x,-a / 2 , a / 2 ) ) #  normalization with minimized coefficients

        # Plots
        plt.figure ( ) # Wavefunctions
        plt.title ( "Infinite well" %N)
        plt.plot ( t , fprufin ( t ) / norm ** 0.5 , "r ." , label = "n =%d variational" %n )
        plt.plot ( t , fpozoana ( 1 , t ) , label =( " n =1 , theoretical " ) )
        plt.plot ( t , fpozoana ( 2 , t ) , label =( " n =2 , theoretical " ) )
        plt.plot ( t , fpozoana ( 3 , t ) , label =( " n =3 , theoretical " ) )
        plt.plot ( t , fpozoana ( 4 , t ) , label =( " n =4 , theoretical " ) )
        plt.legend ( loc = "best")
        plt.grid ( )
        plt.xlabel ( "x" )
        plt.ylabel ( "y" )
        EN.append ( et )
        print (EN)
        plt.show ( )
        Epozan = np.ones ( [max(Nc)] ) * Epozoana ( n )
        cont += 1
    
    plt.figure ( ) # Energies
    plt.title ( "Infinite well , state n = %d" %n )
    plt.plot ( Nrep , EN, "rd" , label =( "Variational energy" ) )
    plt.plot ( Nrep , Epozan , "bd" , label = ( "Analytical energy" ) )
    plt.xlabel ( "N" )
    plt.ylabel ( "E" )
    plt.legend ( loc ="best" )
    plt.show ()