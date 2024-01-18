#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:49:16 2023

@author: sambrandt
"""
# INPUTS ######################################################################

# Time (must be contained within the latest GFS run)
year=2024
month=1
day=20
hour=0 # in UTC

# Edges of Domain (in degrees of latitude/longitude)
north=60
south=22
east=-60
west=-130

# Pressure level (in mb, must be a multiple of 50)
plev=500

# LIBRARIES ###################################################################

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from siphon.catalog import TDSCatalog
from xarray.backends import NetCDF4DataStore
import xarray as xr
from scipy.ndimage import gaussian_filter

# FUNCTIONS ###################################################################

def partial(lat,lon,field,wrt):
    # Calculates horizontal gradients on a lat/lon grid
    gradient=np.zeros(np.shape(field))
    if wrt=='x':
        upper=field[:,2::]
        lower=field[:,0:-2]
        dx=111200*np.cos(lat[:,2::]*(np.pi/180))*(lon[0,1]-lon[0,0])
        grad=(upper-lower)/(2*dx)
        gradient[:,1:-1]=grad
        gradient[:,0]=grad[:,0]
        gradient[:,-1]=grad[:,-1]
    if wrt=='y':
        upper=field[2::,:]
        lower=field[0:-2,:]
        dy=111200*(lat[1,0]-lat[0,0])
        grad=(upper-lower)/(2*dy)
        gradient[1:-1,:]=grad
        gradient[0,:]=grad[0,:]
        gradient[-1,:]=grad[-1,:] 
    return gradient

def laplacian(lat,lon,field):
    # Calculates the horizontal laplacian on a lat/lon grid
    return partial(lat,lon,partial(lat,lon,field,'x'),'x')+partial(lat,lon,partial(lat,lon,field,'y'),'y')

def advection(lat,lon,u,v,field):
    # Calculates horizontal advection of a field on a lat/lon grid
    return u*partial(lat,lon,field,'x')+v*partial(lat,lon,field,'y')

# PHYSICAL CONSTANTS ##########################################################

g=9.8 # Acceleration due to gravity in m/s^2
Omega=7.2921*10**-5 # Rotation rate of Earth in rad/s

# DATA RETRIEVAL ##############################################################

# Location of latest GFS run on THREDDS server
url='https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_onedeg/latest.xml'
# Define location of the data
best_gfs=TDSCatalog(url)
best_ds=list(best_gfs.datasets.values())[0]
ncss=best_ds.subset()
# Create a datetime object to specify the output time that you want
valid=datetime(year,month,day,hour)
# Establish a query for the data
query = ncss.query()
# Trim data to location/time of interest
query.lonlat_box(north=north,south=south,east=east,west=west).time(valid)
#query.add_lonlat(value=True)
# Specify that output needs to be in netcdf format
query.accept('netcdf4')
# Specify the variables that you want
query.variables('Geopotential_height_isobaric','u-component_of_wind_isobaric','v-component_of_wind_isobaric')
# Retrieve the data using the info from the query
data=ncss.get_data(query)
data=xr.open_dataset(NetCDF4DataStore(data))
# Retrieve the model's pressure levels
plevs=np.array(list(map(float,ncss.metadata.axes['isobaric']['attributes'][2]['values'])))
# Define indices for the pressure levels of interest
pindex_middle=np.where(plevs==(plev)*100)[0][0]
pindex_upper=np.where(plevs==(plev-100)*100)[0][0]
pindex_lower=np.where(plevs==(plev+100)*100)[0][0]
# Retrieve height grids
z_middle=np.array(data['Geopotential_height_isobaric'][0,pindex_middle,:,:])
z_upper=np.array(data['Geopotential_height_isobaric'][0,pindex_upper,:,:])
z_lower=np.array(data['Geopotential_height_isobaric'][0,pindex_lower,:,:])

# CALCULATIONS ################################################################

# Create lat/lon grids
lat=np.flip(np.arange(south,north+1))
lon=np.arange(west,east+1)
lon,lat=np.meshgrid(lon,lat)
# Calculate planetary vertical vorticity grid
f=2*Omega*np.sin(lat*(np.pi/180))
# Calculate geostrophic wind grids
u_g_upper=-(g/f)*partial(lat,lon,z_upper,'y')
v_g_upper=(g/f)*partial(lat,lon,z_upper,'x')
u_g_lower=-(g/f)*partial(lat,lon,z_lower,'y')
v_g_lower=(g/f)*partial(lat,lon,z_lower,'x')
# Calculate thermal wind grids
u_thermal=((u_g_upper-u_g_lower)/-20000)
v_thermal=((v_g_upper-v_g_lower)/-20000)
# Calculate geostrophic vertical vorticity grid
vort_g=(g/f)*laplacian(lat,lon,z_middle)
# Calculates Trenberth approximated forcing for QG vertical motion
trenberth=advection(lat,lon,u_thermal,v_thermal,vort_g)

# PLOTTING ####################################################################

# Create geographic axis
fig,ax=plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(),'adjustable': 'box'},dpi=1000)
# Add country borders
ax.add_feature(cfeature.COASTLINE.with_scale('50m'),edgecolor='gray',linewidth=0.5)
# Add state borders
ax.add_feature(cfeature.STATES.with_scale('50m'),edgecolor='gray',linewidth=0.5)
# Filled contour plot of Trenberth forcing
pcm1=ax.contourf(lon,lat,gaussian_filter(trenberth,sigma=4),1e-14*np.delete(np.arange(-10.5,11.5),20),cmap='BrBG',extend='both',alpha=0.75)
# Contour plot of geopotential height at plev
pcm=ax.contour(lon,lat,gaussian_filter(z_middle,sigma=2),np.arange(0,100000,50),linewidths=0.5,colors='black',zorder=3)
# Create colorbar axis
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
# Add colorbar
cbar = plt.colorbar(pcm1,cax=cax)
# Set colorbar tick label size
cbar.ax.tick_params(labelsize=8)
# Figure title
ax.set_title(str(plev)+' mb Trenberth Approx to QG Omega Equation ('+r'$s^{-2} \cdot Pa^{-1}$'+')\n'+str(plev)+' mb Geopotential Height (Contoured Every 50 m)\nGreen = Forcing for Ascent, Brown = Forcing for Descent\nGFS Initialized '+ncss.metadata.time_span['begin'][0:10]+' '+ncss.metadata.time_span['begin'][11:13]+'z, Valid '+str(valid)[0:13]+'z',fontsize=8)
ax.text(west+0.98*(east-west),south+0.02*(north-south),'github.com/SamBrandtMeteo',ha='right',va='bottom',fontsize=6)

###############################################################################