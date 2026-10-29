"""Analytic geometry/scattering tangent of conservative refractive transfer.

Constitutive derivatives (opacity, B and n) are local at fixed pressure.
Differentiate ray weights, turning depths, path resistances and sources;
then solve the same scattering system for dJ. No straight-ray tangent,
frozen ray geometry, finite difference of nearly equal total fluxes, or
per-temperature-column full atmosphere/radiation solve is used.
"""
import numpy as np
from numba import njit

from .dq_refractive_finite_volume import operators, ray_solve, validate
from .dq_refractive_ray_bundles import bundles


@njit(cache=False)
def known_response(mass,k,index,dk,dindex,source,mean,direct,dbottom,angles,weights):
    """Tangent forcing excluding the common scattering feedback through dJ.

    Compile against the current imported geometry; do not reuse a disk
    cache that cannot track edits to that separate research module.
    """
    size=len(mass);cells=size-1
    dj=np.zeros((cells,size));df=np.zeros((size,size));dv=np.zeros((cells,size))
    for first,last,depth,g,dd,dg,weight,dweight in bundles(
            mass,k,index,dk,dindex,angles,weights,True):
        count=last-first+1
        base_rhs=np.zeros((count+1,1));base_rhs[:-1,0]=depth*source[first:last+1]
        base_rhs[-1,0]=source[-1]
        p,v=ray_solve(depth,g,base_rhs)
        rhs=np.zeros((count+1,size))
        for j in range(count):
            i=first+j
            for col in range(size):rhs[j,col]=dd[j,col]*(source[i]-p[j,0])
            rhs[j,i]+=depth[j]*direct[i]
        for col in range(size):rhs[0,col]-=dg[0,col]*p[0,0]
        for j in range(1,count):
            jump=v[j,0]/g[j]
            for col in range(size):
                rhs[j-1,col]+=dg[j,col]*jump
                rhs[j,col]-=dg[j,col]*jump
        if g[-1]>0:
            for col in range(size):rhs[-2,col]+=dg[-1,col]*(v[-1,0]/g[-1])
        rhs[-1,-1]=dbottom
        dp,dflux=ray_solve(depth,g,rhs)
        for col in range(size):dflux[0,col]+=dg[0,col]*p[0,0]
        for j in range(1,count+1):
            if g[j]>0:
                for col in range(size):dflux[j,col]+=dg[j,col]*(v[j,0]/g[j])
        for j in range(count):
            i=first+j
            for col in range(size):
                volume_derivative=dweight[col]*depth[j]+weight*dd[j,col]
                dv[i,col]+=volume_derivative
                dj[i,col]+=weight*depth[j]*dp[j,col]+volume_derivative*(p[j,0]-mean[i])
        for j in range(count+1):
            for col in range(size):df[first+j,col]+=4*np.pi*(weight*dflux[j,col]+dweight[col]*v[j,0])
    return dj,df,dv


def integrated_response(mass,wave,absorption,scattering,planck,index,
        dabsorption,dscattering,dplanck,dindex,*,n_angle=4,progress=None):
    mass,(a,s,b,n),angles,weights=validate(mass,absorption,scattering,planck,index,n_angle)
    wave=np.asarray(wave,float)
    derivatives=tuple(np.asarray(x,float) for x in (dabsorption,dscattering,dplanck,dindex))
    if (wave.shape!=(len(b),) or len(wave)<2 or np.any(~np.isfinite(wave))
        or np.any(wave<=0) or np.any(np.diff(wave)<=0)
        or any(x.shape!=b.shape or np.any(~np.isfinite(x)) for x in derivatives)):
        raise ValueError('Ordered wavelength grid and finite local constitutive derivatives required')
    da,ds,db,dn=derivatives
    ww=np.diff(wave,prepend=wave[0],append=wave[-1]);ww=.5*(ww[:-1]+ww[1:])
    size=len(mass);cells=size-1;ii=np.arange(cells)
    flux_j=np.zeros((size,size));energy_j=np.zeros((cells,size));emission_j=np.zeros_like(energy_j)
    for iw in range(len(wave)):
        k=a[iw]+s[iw];dk=da[iw]+ds[iw]
        eps=a[iw,:-1]/k[:-1];deps=(da[iw,:-1]-eps*dk[:-1])/k[:-1]
        lam,fl,vol=operators(mass,k,n[iw],angles,weights)
        matrix=np.eye(cells)-lam[:,:-1]*(1-eps)[None,:]
        mean=np.linalg.solve(matrix,lam[:,:-1]@(eps*b[iw,:-1])+lam[:,-1]*b[iw,-1])
        source=np.r_[eps*b[iw,:-1]+(1-eps)*mean,b[iw,-1]]
        direct=eps*db[iw,:-1]+deps*(b[iw,:-1]-mean)
        known_j,known_f,dvol=known_response(mass,k,n[iw],dk,dn[iw],source,mean,
            direct,db[iw,-1],angles,weights)
        dj=np.linalg.solve(matrix,known_j/vol[:,None])
        df=known_f+fl[:,:-1]@((1-eps)[:,None]*dj)
        de=4*np.pi*eps[:,None]*(dvol*(mean-b[iw,:-1])[:,None]+vol[:,None]*dj)
        de[ii,ii]+=4*np.pi*vol*(deps*(mean-b[iw,:-1])-eps*db[iw,:-1])
        dem=4*np.pi*(eps*b[iw,:-1])[:,None]*dvol
        dem[ii,ii]+=4*np.pi*vol*(deps*b[iw,:-1]+eps*db[iw,:-1])
        flux_j+=ww[iw]*df;energy_j+=ww[iw]*de;emission_j+=ww[iw]*dem
        if progress is not None and (iw+1)%1000==0:progress(iw+1,len(wave))
    return flux_j,energy_j,emission_j
