import numpy as np
import cv2
import math


class Snake:
   
    # Constants
    MIN_DISTANCE = 5    # The minimum distance between two points to consider them overlaped
    MAX_DISTANCE = 50    # The maximum distance to insert another point into the spline
    SEARCH_KERNEL_SIZE = 5             # The size of the search kernel.

    # Members
    image = None        # The source image.
    gray = None         # The image in grayscale.
    binary = None       # The image in binary (threshold method).
    gradientX = None    # The gradient (sobel) of the image relative to x.
    gradientY = None    # The gradient (sobel) of the image relative to y.
    blur = None
    width = -1          # The image width.
    height = -1         # The image height.
    points = None       # The list of points of the snake.
    n_starting_points = 50       # The number of starting points of the snake.
    snake_length = 0    # The length of the snake (euclidean distances).
    closed = True       # Indicates if the snake is closed or open.
    alpha = 4         # The weight of the uniformity energy.
    beta = 2          # The weight of the curvature energy.
    gamma = 3        # The weight to the edge energy.


    def __init__( self, image = None, closed = True ):
      
        self.image = image
        
        self.height,self.width  = image.shape[0], image.shape[1]

        
        self.gray = cv2.cvtColor( self.image, cv2.COLOR_RGB2GRAY )
        self.binary = cv2.adaptiveThreshold( self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2 )
        self.gradientX = cv2.Sobel( self.gray, cv2.CV_64F, 1, 0, ksize=5 )
        self.gradientY = cv2.Sobel( self.gray, cv2.CV_64F, 0, 1, ksize=5 )
        
        half_width = math.floor( self.width / 2 )
        half_height = math.floor( self.height / 2 )

        n = self.n_starting_points
        radius = half_width if half_width < half_height else half_height
        self.points = [ np.array([  half_width + math.floor( math.cos( 2 * math.pi / n * x ) * radius ),
                                    half_height + math.floor( math.sin( 2 * math.pi / n * x ) * radius ) ])
                                    for x in range( 0, n )]
       

    def visualize( self ):
  
        img = self.image.copy()

        point_color = ( 0, 255, 255 )     # BGR RED
        line_color = ( 128, 0, 0 )      # BGR half blue
        thickness = 2                   # Thickness of the lines and circles

        n_points = len( self.points )
        for i in range( 0, n_points - 1 ):
            cv2.line( img, tuple( self.points[ i ] ), tuple( self.points[ i + 1 ] ), line_color, thickness )

        if self.closed:
            cv2.line(img, tuple( self.points[ 0 ] ), tuple( self.points[ n_points-1 ] ), line_color, thickness )

        [ cv2.circle( img, tuple( x ), thickness, point_color, -1) for x in self.points ]

        return img

    def dist( a, b ):
        return np.sqrt( np.sum( ( a - b ) ** 2 ) )

    def normalize( kernel ):

        abs_sum = np.sum( [ abs( x ) for x in kernel ] )
        return kernel / abs_sum if abs_sum != 0 else kernel

    def get_length(self):
    
        n_points = len(self.points)
        if not self.closed:
            n_points -= 1

        return np.sum( [ Snake.dist( self.points[i], self.points[ (i+1)%n_points  ] ) for i in range( 0, n_points ) ] )

    def Sobel(self,img,n):
        maskX = [[2,2,2,2,2],[1,1,1,1,1],[0,0,0,0,0],[-1,-1,-1,-1,-1],[-2,-2,-2,-2,-2]]
        maskY = [[2,1,0,-1,-2],[2,1,0,-1,-2],[2,1,0,-1,-2],[2,1,0,-1,-2],[2,1,0,-1,-2]]
        R,C = img.shape
        newImage = np.zeros((R+n-1,C+n-1))
        newImage = np.zeros((R+n-1,C+n-1))
        self.gradientX =[]
        self.gradientY = []
        
        for i in range( 1 , R-4 ):
            for j in range( 1 , C-4 ):
                    S1 = np.sum(np.multiply(maskX,img[i:i+n,j:j+n]))
                    S2 = np.sum(np.multiply(maskY,img[i:i+n,j:j+n]))
                    self.gradientX.append(S1)
                    self.gradientY.append(S2)
                    newImage[i+1,j+1] = np.sqrt(np.power(S1,2)+np.power(S2,2))
        
        newImage *= 255.0 / newImage.max()
        
        return(newImage)

    def f_uniformity( self, p, prev ):
     
        avg_dist = self.snake_length / len( self.points )
     
        un = Snake.dist( prev, p )

        dun = abs( un - avg_dist )

        return dun**2


    
    def curvature( self, p, prev, next ):

        ux = p[0] - prev[0]
        uy = p[1] - prev[1]
        un = math.sqrt( ux**2 + uy**2 )

        vx = p[0] - next[0]
        vy = p[1] - next[1]
        vn = math.sqrt( vx**2 + vy**2 )

        if un == 0 or vn == 0:
            return 0

        cx = float( vx + ux )  / ( un * vn )
        cy = float( vy + uy ) / ( un * vn )

        cn = cx**2 + cy**2

        return cn

    def edge (self, p ):

        if p[0] < 0 or p[0] >= self.width or p[1] < 0 or p[1] >= self.height:
           
            return np.finfo(np.float64).max

        return -( self.gradientX[ p[1] ][ p[0] ]**2 + self.gradientY[ p[1] ][ p[0] ]**2  )


    def remove_overlaping_points( self ):

        snake_size = len( self.points )

        for i in range( 0, snake_size ):
            for j in range( snake_size-1, i+1, -1 ):
                if i == j:
                    continue

                curr = self.points[ i ]
                end = self.points[ j ]

                dist = Snake.dist( curr, end )

                if dist < self.MIN_DISTANCE:
                    remove_indexes = range( i+1, j ) if (i!=0 and j!=snake_size-1) else [j]
                    remove_size = len( remove_indexes )
                    non_remove_size = snake_size - remove_size
                    if non_remove_size > remove_size:
                        self.points = [ p for k,p in enumerate( self.points ) if k not in remove_indexes ]
                    else:
                        self.points = [ p for k,p in enumerate( self.points ) if k in remove_indexes ]
                    snake_size = len( self.points )
                    break


    def add_missing_points( self ):
       
        snake_size = len( self.points )
        for i in range( 0, snake_size ):
            prev = self.points[ ( i + snake_size-1 ) % snake_size ]
            curr = self.points[ i ]
            next = self.points[ (i+1) % snake_size ]
            next2 = self.points[ (i+2) % snake_size ]

            if Snake.dist( curr, next ) > self.MAX_DISTANCE:
                c0 = 0.125 / 6.0
                c1 = 2.875 / 6.0
                c2 = 2.875 / 6.0
                c3 = 0.125 / 6.0
                x = prev[0] * c3 + curr[0] * c2 + next[0] * c1 + next2[0] * c0
                y = prev[1] * c3 + curr[1] * c2 + next[1] * c1 + next2[1] * c0

                new_point = np.array( [ math.floor( 0.5 + x ), math.floor( 0.5 + y ) ] )

                self.points.insert( i+1, new_point )
                snake_size += 1


    def step( self ):
 
        changed = False
        self.snake_length = self.get_length()
        new_snake = self.points.copy()
        search_kernel_size = ( self.SEARCH_KERNEL_SIZE, self.SEARCH_KERNEL_SIZE )
        hks = math.floor( self.SEARCH_KERNEL_SIZE / 2 ) # half-kernel size
        e_uniformity = np.zeros( search_kernel_size )
        e_curvature = np.zeros( search_kernel_size )
        e_edge = np.zeros( search_kernel_size )
 

        for i in range( 0, len( self.points ) ):
            curr = self.points[ i ]
            prev = self.points[ ( i + len( self.points )-1 ) % len( self.points ) ]
            next = self.points[ ( i + 1 ) % len( self.points ) ]


            for dx in range( -hks, hks ):
                for dy in range( -hks, hks ):
                    p = np.array( [ curr[0] + dx, curr[1] + dy] )

                    # Calculates the energy functions on p
                    e_uniformity[ dx + hks ][ dy + hks ] = self.f_uniformity( p, prev )
                    e_curvature[ dx + hks ][ dy + hks ] = self.curvature( p, prev, next )
                    e_edge[ dx + hks ][ dy + hks ] = self.edge( p )
              
            # Normalizes energies
            e_uniformity = Snake.normalize( e_uniformity )
            e_curvature = Snake.normalize( e_curvature )
            e_edge = Snake.normalize( e_edge )

            e_sum = self.alpha * e_uniformity \
                    + self.beta * e_curvature \
                    + self.gamma * e_edge 
                    

            # Searches for the point that minimizes the sum of energies e_sum
            emin = np.finfo(np.float64).max
            x,y = 0,0
            for dx in range( -hks, hks ):
                for dy in range( -hks, hks ):
                    if e_sum[ dx + hks ][ dy + hks ] < emin:
                        emin = e_sum[ dx + hks ][ dy + hks ]
                        x = curr[0] + dx
                        y = curr[1] + dy
           
            # Boundary check
            x = 1 if x < 1 else x
            x = self.width-2 if x >= self.width-1 else x
            y = 1 if y < 1 else y
            y = self.height-2 if y >= self.height-1 else y

            # Check for changes
            if curr[0] != x or curr[1] != y:
                changed = True

            new_snake[i] = np.array( [ x, y ] )

        self.points = new_snake

        # Post threatment to the snake, remove overlaping points and
        # add missing points
        self.remove_overlaping_points()
        self.add_missing_points()

        return changed


    def set_alpha( self, x ):
        self.alpha = x / 100

    def set_beta( self, x ):

        self.beta = x / 100

    def set_gamma( self, x ):
        self.gamma = x / 100
 
   