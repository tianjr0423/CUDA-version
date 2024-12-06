#include<bits/stdc++.h> 
#include <cuda_runtime.h>
#include <time.h>
#include <stdio.h>
using namespace std;

#define pver 51
#define pverp 52
#define pcnst 25
#define pcols 32768//1024
#define ncol 32768//1024
int delt=1.0;
int PERGRO;//!==0为false，==1为true，替换编译判断,全局变量
double latvap=15.0;
double cpair=10.0;
bool no_deep_pbl=false;
double c1 = 6.112;
double c2 = 17.67;
double c3 = 243.5;
double eps1 = 1.0e-8;
double  tiedke_add = 0.5;
double tfreez = 1.0;
double cpliq=1.0;
double latice = 1.0;
double rgas = 1.4;
double zmconv_dmpdz=1.0;
double c0_ocn=1.0;
double c0_lnd=1.0;
double tau=3600.0;
double capelmt =0.0; //70.0;
int msg=0; 
double grav=9.8;
//设置线程
#define TPBN 8
#define TPBL 32

//main    
__device__ double qh[pcnst][pver][pcols];
__device__ double t[pver][pcols];  
__device__ double pap[pver][pcols];
__device__ double paph[pverp][pcols];
__device__ double dpp[pver][pcols];    
__device__ double zm[pver][pcols];
__device__ double geos[pcols];
__device__ double zi[pverp][pcols];
__device__ double pblh[pcols];
__device__ double tpert[pcols];
__device__ double landfrac[pcols];
__device__ double qtnd[pver][pcols];   
__device__ double heat[pver][pcols];       
__device__ double mcon[pverp][pcols];
__device__ double dlf[pver][pcols];  
__device__ double pflx[pverp][pcols];
__device__ double cme[pver][pcols];
__device__ double cape[pcols];      
__device__ double zdu[pver][pcols];
__device__ double rprd[pver][pcols];
__device__ double mu[pver][pcols];
__device__ double eu[pver][pcols];
__device__ double du[pver][pcols];
__device__ double md[pver][pcols];
__device__ double ed[pver][pcols];
__device__ double dp[pver][pcols];  
__device__ double dsubcld[pcols];
__device__ double jctop[pcols];
__device__ double jcbot[pcols];
__device__ double prec[pcols];
__device__ double rliq[pcols];
__device__ double jt[pcols];               
__device__ double maxg[pcols];               
__device__ int ideep[pcols];                    
__device__ double ql[pver][pcols];                
__device__ int lengath;
__device__ double zs[pcols];
__device__ double dlg[pver][pcols];    
__device__ double pflxg[pverp][pcols];
__device__ double cug[pver][pcols];  
__device__ double evpg[pver][pcols];   
__device__ double mumax[pcols];              
__device__ double pblt[pcols];
//--------------------------
__device__ double q[pver][pcols];
__device__ double p[pver][pcols];
__device__ double z[pver][pcols];
__device__ double s[pver][pcols];
__device__ double tp[pver][pcols];//          
__device__ double zf[pverp][pcols];
__device__ double pf[pverp][pcols];//      
__device__ double qstp[pver][pcols];//           
__device__ double tl[pcols];//                 
__device__ int lcl[pcols];//                
__device__ int lel[pcols];//                 
__device__ int lon[pcols];//                
__device__ int maxi[pcols];//                 
__device__ int indexx[pcols];//
//__device__ double precip;
__device__ double qg[pver][pcols];//         
__device__ double tg[pver][pcols];//       
__device__ double pg[pver][pcols];//       
__device__ double zg[pver][pcols];//              
__device__ double sg[pver][pcols];//         
__device__ double tpg[pver][pcols];//         
__device__ double zfg[pverp][pcols];//(pcols,pver+1)         
__device__ double qstpg[pver][pcols];//[pver][pcols];          
__device__ double ug[pver][pcols];//[pver][pcols];           
__device__ double vg[pver][pcols];//[pver][pcols];              
__device__ double cmeg[pver][pcols];//[pver][pcols]; 
__device__ double rprdg[pver][pcols];//[pver][pcols];          
__device__ double capeg[pcols];//              
__device__ double tlg[pcols];//                
__device__ double landfracg[pcols];//
__device__ int lclg[pcols];//      
__device__ int lelg[pcols];//
__device__ double dqdt[pver][pcols];//[pver][pcols];            
__device__ double dsdt[pver][pcols];//[pver][pcols];            
__device__ double sd[pver][pcols];//[pver][pcols];             
__device__ double qd[pver][pcols];//[pver][pcols];             
__device__ double mc[pver][pcols];//[pver][pcols];              
__device__ double qhat[pver][pcols];//[pver][pcols];           
__device__ double qu[pver][pcols];//[pver][pcols];              
__device__ double su[pver][pcols];//[pver][pcols];              
__device__ double hu[pver][pcols];//[pver][pcols];             
__device__ double qs[pver][pcols];//[pver][pcols];             
__device__ double shat[pver][pcols];//[pver][pcols];           
__device__ double hmn[pver][pcols];//[pver][pcols];            
__device__ double hsat[pver][pcols];//[pver][pcols];           
__device__ double qlg[pver][pcols];//[pver][pcols]; 
__device__ double dudt[pver][pcols];//[pver][pcols];            
__device__ double dvdt[pver][pcols];//[pver][pcols];            
__device__ double mb[pcols];//                 
//=======================boyan===========================
__device__ int mx[pcols];   
__device__ double capeten[5][pcols];   //! provisional value of cape
__device__ double tv[pver][pcols];        
__device__ double tpv[pver][pcols];       
__device__ double buoy[pver][pcols]; 
__device__ double a1[pcols]; 
__device__ double a2[pcols]; 
__device__ double estp[pcols]; 
__device__ double pl[pcols]; 
__device__ double plexp[pcols]; 
__device__ double hmax[pcols]; 
__device__ double y[pcols]; 
__device__ int knt[pcols]; 
__device__ int lelten[5][pcols];

//=======================cldprp===========================
__device__ double qst[pver][pcols]; 
__device__ double f[pver][pcols];//[pver][pcols];
__device__ double ggamma[pver][pcols];//[pver][pcols];
__device__ double qds[pver][pcols];//[pver][pcols];
__device__ double cu[pver][pcols];//[pver][pcols];
__device__ double u[pver][pcols];//[pver][pcols];//[pver][pcols];         //! zonal velocity of env
__device__ double v[pver][pcols];//[pver][pcols];//[pver][pcols];        // ! merid. velocity of env 
__device__ double dz[pver][pcols];//[pver][pcols];
__device__ double iprm[pver][pcols];//[pver][pcols];
__device__ double hd[pver][pcols];//[pver][pcols];
__device__ double hdd[pver][pcols];//[pver][pcols];
__device__ double eps[pver][pcols];//[pver][pcols];
__device__ double k1[pver][pcols];//[pver][pcols];
__device__ double i2[pver][pcols];//[pver][pcols];
__device__ double ihat[pver][pcols];//[pver][pcols];
__device__ double i3[pver][pcols];//[pver][pcols];
__device__ double idag[pver][pcols];//[pver][pcols];
__device__ double i4[pver][pcols];//[pver][pcols];
__device__ double qsthat[pver][pcols];//[pver][pcols];
__device__ double hsthat[pver][pcols];//[pver][pcols];
__device__ double gamhat[pver][pcols];//[pver][pcols];
__device__ double evp[pver][pcols];//[pver][pcols];
__device__ double c0mask[pcols];//[pcols]
__device__ double hmin[pcols];//[pcols]
__device__ double expdif[pcols];//[pcols]
__device__ double expnum[pcols];//[pcols]
__device__ double ftemp[pcols];//[pcols]
__device__ double eps0[pcols];//[pcols]
__device__ double rmue[pcols];//[pcols]
__device__ double zuef[pcols];//[pcols]
__device__ double zdef[pcols];//[pcols]
__device__ double epsm[pcols];//[pcols]
__device__ double ratmjb[pcols];//[pcols]
__device__ double est[pcols];//[pcols]
__device__ double totpcp[pcols];//[pcols]
__device__ double totevp[pcols];//[pcols]
__device__ double alfa[pcols];
__device__ int jb[pcols];

__device__ double  dsubb[pcols];

__global__ void zm_convr1()
{int i,k;

double grav=9.8;
double cpres = 10.0;

    k = blockDim.x * blockIdx.x + threadIdx.x;
    i = blockDim.y * blockIdx.y + threadIdx.y;
    if((k>=0 and k<pver) and (i>=0 and i<ncol))
    {
    qtnd[k][i]=0;
    heat[k][i]=0;
    mcon[k][i]=0;
   // printf("qtnd=%E   \n", qtnd[k][i]);
    }

    if(i>=0 and i<ncol)
    {
     prec[i]=0;   
    }

     if((k>=0 and k<pver) and (i>=0 and i<ncol))
    {
    dqdt[k][i]=0;
    dsdt[k][i]=0;
    dudt[k][i]=0;
    dvdt[k][i]=0;
    pflx[k][i]=0;
    pflxg[k][i]=0;
    cme[k][i]=0;
    rprd[k][i]=0;
    zdu[k][i]=0;
    ql[k][i]=0;
    qlg[k][i]=0;
    dlf[k][i]=0;
    dlg[k][i]=0;
    //printf("qtnd=%E   \n", qtnd[k][i]);
    }

    if(i>=0 and i<ncol)
    {
    pflx[pver][i] = 0;
    pflxg[pver][i] = 0;   
    }

    if(i>=0 and i<ncol)
    {
    //pblt[i]= pver;
    pblt[i]= pver;
    dsubcld[i]= 0;
    jctop[i]= pver;
    jcbot[i]= 1 ;     
    }

    if(i>=0 and i<ncol)
    {
    zs[i] = geos[i]*grav;
    pf[pver][i]= paph[pver][i]*0.01;
    zf[pver][i] = zi[pver][i]+ zs[i]; 
    }

    if((k>=0 and k<pver) and (i>=0 and i<ncol))
    {
    p[k][i] = pap[k][i]*0.01;
    pf[k][i] = paph[k][i]*0.01;
    z[k][i] = zm[k][i] + zs[i];
    zf[k][i] = zi[k][i] + zs[i]; 
    }

    if((k>=0 and k<pver-1) and (i>=0 and i<ncol))
    {
    if(abs(z[k][i]-zs[i]-pblh[i])<((zf[k][i]-zf[k+1][i])*0.5))
    {
    	pblt[i] = k;
	  } 
    }

    if((k>=0 and k<pver) and (i>=0 and i<ncol))
    {
    q[k][i] = (i+1)*(k+1);//(i,k,1)
    s[k][i] = t[k][i] + (grav/cpres)*z[k][i];
    tp[k][i]=0.0;
    shat[k][i]= s[k][i];
    qhat[k][i]= q[k][i];
   // printf("=%d %d  %f %f  \n",i+1,k+1, shat[k][i], qhat[k][i]);
    }

    if(i>=0 and i<ncol)
    {
    capeg[i]= 0;
    lclg[i]= 1;
    lelg[i]= pver;
    maxg[i] = 1;
    tlg[i]= 400;
    dsubcld[i] = 0; 
    }	

}

__device__ int jlcl[pcols];//
__device__ int jj0[pcols];//                 ! wg detrainment initiation level index.
__device__ int jd[pcols];//                 ! wg downdraft initiation level index.
__device__ double hmnn[pcols];

__global__ void buoyan()
{

double grav=9.8;
double cp=10.0;
double e; 
int i;
int k;
int msg=0;
int n;
double rd=1.4;
double rl=15.0;
double rhd;	
double eps1 = 1.0e-8;
i = blockDim.y * blockIdx.y + threadIdx.y;
k = blockDim.x * blockIdx.x + threadIdx.x;

if((k>=0 and k<5) and (i>=0 and i<ncol))
{
  lelten[k][i]= pver;
  capeten[k][i]= 0;
}

if(i>=0 and i<ncol)
{
  lon[i]= pver-1;
  knt[i]= 0;
  lel[i]= pver-1;
  mx[i]= lon[i];
  cape[i]= 0;
  hmax[i]= 0;
}

if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
  tp[k][i]=t[k][i];
  qstp[k][i]=q[k][i];
}

if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
  tv[k][i]=t[k][i]*(1+1.608*q[k][i])/(1+q[k][i]);
	tpv[k][i]=tv[k][i];	   
  buoy[k][i]=0;
}

if (i>=0 and i<ncol)
{
     hmnn[i] = cp*t[msg][i] + grav*z[msg][i] + rl*q[msg][i];   
     rhd = (hmnn[i] - hmax[i])/(hmnn[i] + hmax[i]);   
   	 hmax[i] = hmnn[i];
     mx[i] = msg;   
    // printf("=%d %d  %f %f  \n",i+1,k+1, hmnn[i], mx[i]);
}

if (i>=0 and i<ncol)
{
  lcl[i]=mx[i];
  e= p[mx[i]][i]*q[mx[i]][i]/ (eps1+q[mx[i]][i]);
  tl[i] = 2840/ (3.5*log(t[mx[i]][i])-log(e)-4.805) + 55;
  plexp[i] = (1/ (0.2854* (1-0.28*q[mx[i]][i])));
  pl[i] = p[mx[i]][i]*pow(tl[i]/t[mx[i]][i],plexp[i]) ; //(tl[i]/t[mx[i]][i])**plexp[i];  

}

double tfreez = 1.0;
double c1 = 6.112;
double c2 = 17.67;
double c3 = 243.5;
double  tiedke_add = 0.5;

if((k>=0 and k<pver-1) and (i>=0 and i<ncol))
{
                   
            tv[k][i] = t[k][i]* (1+1.608*q[k][i])/ (1+q[k][i]);  
            tp[k][i] = tp[k][i]*pow((p[k][i]/p[k+1][i]),(0.2854* (1-0.28*qstp[k][i])));
            estp[i] = c1*exp((c2* (tp[k][i]-tfreez))/((tp[k][i]-tfreez)+c3));    
            qstp[k][i] = eps1*estp[i]/ (p[k][i]-estp[i]);
            a1[i] = cp/rl + qstp[k][i]* (1+qstp[k][i]/eps1)*rl*eps1/pow(rd*tp[k][i],2) ;      //(rd*tp[k][i]**2);
            a2[i] = .5* (qstp[k][i]* (1+2/eps1*qstp[k][i])*(1 +qstp[k][i]/eps1)*pow(eps1,2)*rl*rl/(pow(rd,2)*
			      pow(tp[k][i],4))-qstp[k][i]*(1+qstp[k][i]/eps1)*2*eps1*rl/(rd*pow(tp[k][i],3)));
            a1[i] = 1/a1[i];
            a2[i] = -a2[i]*pow(a1[i],3);//a1[i]**3;
            y[i] = qstp[k+1][i] - qstp[k][i];
             tp[k][i] = tp[k][i] + a1[i]*y[i] + a2[i]*pow(y[i],2);//y[i]**2;
             qstp[k][i] = eps1*estp[i]/ (p[k][i]-estp[i])    ;      
             tpv[k][i] = (tp[k][i]+tpert[i])* (1+1.608*qstp[k][i])/(1+q[mx[i]][i]);
            buoy[k][i] = tpv[k][i] - tv[k][i] + tiedke_add;
}

if((k>=1 and k<pver) and (i>=0 and i<ncol))
{
 if(k<lcl[i] and pl[i]>=0)
    {
    knt[i] = min(knt[i] + 1,5);
    lelten[knt[i]][i]= k;
    //	cout<<2;
	} 
}

if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
  for(int n=0;n<5;n++)
  {
    if(pl[i]>=0 and  k <= mx[i] and k > lelten[n][i])
    {
    	capeten[n][i] = capeten[n][i] + rd*buoy[k][i]*log(pf[k+1][i]/pf[k][i]);
    	//cout<<2;
	}
  }
}

if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
  for(int n=0;n<5;n++)
  {
    if(capeten[n][i] > cape[i])
    {
    cape[i] = capeten[n][i];
    lel[i] = lelten[n][i];	
    //cout<<3;
  }
}}

if (i>=0 and i<ncol)
{
    cape[i]= max(cape[i],1.0); 
}

}

__global__ void zm_convr2()
{
  int i,k;
    k = blockDim.x * blockIdx.x + threadIdx.x;
    i = blockDim.y * blockIdx.y + threadIdx.y;

double capelmt =0.0;

if (i>=0 and i<ncol)
{
     mumax[i] =0;    
}

if (i>=0 and i<lengath)
{
     int ss; 
     ss=indexx[i];
     ideep[i]=ss; 
}

if((k>=0 and k<pver) and (i>=0 and i<lengath))
{
    dp[k][i] = 0.01*dpp[k][ideep[i]];
    qg[k][i] = q[k][ideep[i]];
    tg[k][i] = t[k][ideep[i]];
    pg[k][i] = p[k][ideep[i]];
    zg[k][i] = z[k][ideep[i]];
    sg[k][i] = s[k][ideep[i]];
    tpg[k][i] = tp[k][ideep[i]];
    zfg[k][i] = zf[k][ideep[i]];
    qstpg[k][i] = qstp[k][ideep[i]];
    ug[k][i] = 0;
    vg[k][i] = 0;
    //printf("%d %d %f\n",i,k,dp[k][i]); 
}

if (i>=0 and i<lengath)
{
zfg[pver][i] = zf[pver][ideep[i]];
}

if (i>=0 and i<lengath)
{
    capeg[i] = cape[ideep[i]];
    lclg[i] = lcl[ideep[i]];
    lelg[i] = lel[ideep[i]];
    maxg[i] = maxi[ideep[i]];
    tlg[i] = tl[ideep[i]];
    landfracg[i] = landfrac[ideep[i]];
   // printf("%d %f\n",i+1,maxg[i]); 
}

if(i>=0 and i<ncol)
    {
    dsubcld[i] = 0; 
    }	


if (i>=0 and i<lengath)
{
    for(int m=0;m<pver;m++)
    {
     if (m >= maxg[i])
	   {
	    dsubb[i] = dsubb[i] + dp[m][i];	
     } 
    }
}

   if(i>=0 and i<ncol)
    {
    dsubcld[i] = dsubb[i]; 
    }

if((k>=1 and k<pver) and (i>=0 and i<lengath))
{
   double sdifr = 0;
   double qdifr = 0;
    if (sg[k][i] > 0 or sg[k-1][i] > 0) 
        sdifr = abs((sg[k][i]-sg[k-1][i])/max(sg[k-1][i],sg[k][i]));
    if (qg[k][i] > 0 or qg[k-1][i] > 0) 
        qdifr = abs((qg[k][i]-qg[k-1][i])/max(qg[k-1][i],qg[k][i]));
    if (sdifr > 1.0E-6) 
        shat[k][i] = log(sg[k-1][i]/sg[k][i])*sg[k-1][i]*sg[k][i]/(sg[k-1][i]-sg[k][i]);
    else
        shat[k][i] = 0.5* (sg[k][i]+sg[k-1][i]);
    if (qdifr > 1.0E-6) 
        qhat[k][i] = log(qg[k-1][i]/qg[k][i])*qg[k-1][i]*qg[k][i]/(qg[k-1][i]-qg[k][i]);
    else
        qhat[k][i] = 0.5* (qg[k][i]+qg[k-1][i]);
}

}


__global__ void cldprp()
{
int k,i;
double rl = 15.0; 
int il2g=ncol;
double rd=1.4;
double cp=10.0;
int msg=0;
double c1 = 6.112;
double c2 = 17.67;
double c3 = 243.5;
double eps1 = 1.0e-8;
double grav=9.8;
double tfreez = 1.0;
double tiedke_add = 0.5;

double ql1;
double tu;
double estu;
double qstu;
double small;
double mdt;
int khighest;
int klowest;
int kount;

double c0_ocn=1.0;
double c0_lnd=1.0;
k = blockDim.x * blockIdx.x + threadIdx.x;
i = blockDim.y * blockIdx.y + threadIdx.y;

if(i>=0 and i<ncol)
{
  jb[i] =maxg[i];	 
}	

if(i>=0 and i<ncol)
{
  ftemp[i] = 0;
  expnum[i] = 0;
  expdif[i] = 0;
  c0mask[i]  = c0_ocn * (1-landfrac[i]) +   c0_lnd * landfrac[i]; 
   //printf("%d %f\n",i+1,c0mask[i]); 

}	

if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
dz[k][i] = zf[k][i] - zf[k+1][i];
//printf("%d %d %f\n",i+1,k+1,dz[k][i]);
}

if(i>=0 and i<ncol)
{
  pflx[0][i]=0; 
  eps0[i]=0.0001;
  //printf("%d %f\n",i+1,eps0[i]);
}

if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
    k1[k][i] = 0;
    i2[k][i] = 0;
    i3[k][i] = 0;
    i4[k][i] = 0;
    mu[k][i] = 0;
    f[k][i] = 0;
    eps[k][i] =  0.0001;
    eu[k][i] = 0;
    du[k][i] = 0;
    ql[k][i] = 0;
    u[k][i] = 0;

}

if((k>=0 and k<pver) and (i>=0 and i<ncol))
{   
    //qds[k][i] = q[k][i];
    md[k][i] = 0;
    ed[k][i] = 0;
    //sd[k][i] = s[k][i];
    //qd[k][i] = q[k][i];
    //mc[k][i] = 0;
    qu[k][i] = q[k][i];
    su[k][i] = s[k][i];
    est[i] = c1*exp((c2* (t[k][i]-tfreez))/((t[k][i]-tfreez)+c3));

    if ( p[k][i]-est[i] > 0 ) 
        qst[k][i] = eps1*est[i]/ (p[k][i]-est[i]);
    else
        qst[k][i] = 1.0;
            
    ggamma[k][i] = qst[k][i]*(1.0 + qst[k][i]/eps1)*eps1*rl/(rd*pow(t[k][i],2))*rl/cp;
    hmn[k][i] = cp*t[k][i] + grav*z[k][i] + rl*q[k][i];
    hsat[k][i] = cp*t[k][i] + grav*z[k][i] + rl*qst[k][i];
    hu[k][i] = hmn[k][i];
    hd[k][i] = hmn[k][i];
    rprd[k][i] = 0;
    //printf("%d %d %f\n",i+1,k+1,gamma[k][i]); 

}

if(i>=0 and i<ncol)
{
    hsthat[msg][i] = hsat[msg][i];
    qsthat[msg][i] = qst[msg][i];
    gamhat[msg][i] = ggamma[msg][i];
    totpcp[i] = 0;
    totevp[i] = 0; 
}

if((k>=1 and k<pver) and (i>=0 and i<ncol))
{
    if (abs(qst[k-1][i]-qst[k][i]) > 1) 
        qsthat[k][i] = log(qst[k-1][i]/qst[k][i])*qst[k-1][i]*qst[k][i]/ (qst[k-1][i]-qst[k][i]);
    else
        qsthat[k][i] = qst[k][i];

    hsthat[k][i] = cp*shat[k][i] + rl*qsthat[k][i];
    
    if (abs(ggamma[k-1][i]-ggamma[k][i]) > 1) 
        gamhat[k][i] = log(ggamma[k-1][i]/ggamma[k][i])*ggamma[k-1][i]*ggamma[k][i]/ (ggamma[k-1][i]-ggamma[k][i]);
    else
        gamhat[k][i] = ggamma[k][i];
}

if(i>=0 and i<ncol)
{
   jt[i] = pver-1;
}

if(i>=0 and i<ncol)
{
    jt[i] = max(lel[i],1);
    double ss=pver-1;
    jt[i] = ss;
    jd[i] = pver-1;
    jlcl[i] = lel[i];
    hmin[i] = 1.0E6; 
    //printf("%d  %f \n",i+1,jt[i] ); 
}

if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
if (hsat[k][i] <= hmin[i] and k >= jt[i] and k <= jb[i]) 
    {
        hmin[i] = hsat[k][i];   
        jj0[i] = k;
       //printf("%d  \n",i+1 ); 
    }
}

if(i>=0 and i<ncol)
{  
    jj0[i] =pver-1; 
  //printf("j0%d  %d \n",i+1,j0[i] ); 50
}
 
if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
hu[k][i] = hmn[mx[i]][i] + cp*tiedke_add;
su[k][i] = s[mx[i]][i] + tiedke_add;
//printf("%d %d %f %f\n",i+1,k+1,hu[k][i],su[k][i]);
}

if(i>=0 and i<ncol)
{
 hmin[i] = 1.0E6;  
}

if(i>=0 and i<ncol)
{
    expnum[i] = 0;
    ftemp[i] = 0;  
}

if((k>=1 and k<pver) and (i>=0 and i<ncol))
{
       
    expnum[i] = hmn[mx[i]][i] - (hsat[k-1][i]*(zf[k][i]-z[k][i]) + 
                hsat[k][i]* (z[k-1][i]-zf[k][i]))/(z[k-1][i]-z[k][i]);
    ftemp[i] = expnum[i]-k1[k][i];
    f[k][i] = ftemp[i] + i2[k][i]/k1[k][i]*pow(ftemp[i],2)+ 
            (2*pow(i2[k][i],2)-k1[k][i]*i3[k][i])/pow(k1[k][i],2)* 
            pow(ftemp[i],3) + (-5*k1[k][i]*i2[k][i]*i3[k][i]+ 
            5*pow(i2[k][i],3)+pow(k1[k][i],2)*i4[k][i])/ pow(k1[k][i],3)*pow(ftemp[i],4 );
	if(jj0[i]==pver-1)
   {f[k][i]=0;   }                 
    f[k][i] = max(f[k][i],0.0001);
    //printf("%d %d %f \n",i+1,k+1,f[k][i]);
}


if(i>=0 and i<ncol)
{
   //printf("=%d   %f   \n",i+1,eps0[i]);
}	

if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
if (eps0[i] > 0)
	{   zuef[i] = zf[k][i] - zf[jb[i]][i];//[jb[i]][i];
      rmue[i] = (1/eps0[i])* (exp(eps[k][i]*zuef[i])-1)/zuef[i];
      mu[k][i] = (1/eps0[i])* (exp(eps[k][i]*zuef[i])-1)/zuef[i];
      eu[k][i] = (rmue[i]-mu[k][i])/dz[k][i];
      du[k][i] = mu[k][i]/dz[k][i];
      hu[k][i] = dz[k][i]/mu[k][i]* (eu[k][i]*hmn[k][i]- du[k][i]*hsat[k][i]);
      //printf("=%d %d  %f  %f %f %f %f %f %f\n",i+1,k+1,hu[k][i],dz[k][i],mu[k][i],eu[k][i],hmn[k][i],du[k][i],hsat[k][i]);
      //printf("=%d %d  %f  %f %f\n",i+1,k+1,rmue[i],mu[k][i] ,eu[k][i]);
      //printf("=%d %d  %f  \n",i+1,k+1,du[k][i]);
  }
  
}

if(i>=0 and i<ncol)
{
  eu[pver-1][i] = 0.0001;
  du[pver-1][i] = -1.0025;
}

if(i>=0 and i<ncol)
{   int kk=pver;

    alfa[i] = 0.1;
    zdef[i] = 0.01;
    jt[i] = -1;
    jd[i] =0;
    //hd[jd[i]][i] = hmn[jd[i]-1][i];
    epsm[i] = i+1;
}

if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
md[k][i] = -alfa[i]/ (2*eps0[i])*(exp(2*epsm[i]*zdef[i])-1)/zdef[i];
}

small=1.0e-20;
if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
  if(eps0[i] > 0)
	{
	  ed[k][i] = (md[k][i])/dz[k][i];
    //mdt = min(md[k][i],-small);
    //hd[k][i] = (md[k][i] - dz[k][i]*ed[k][i]*hmn[k][i])/mdt	;

	}
}

if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
  if(eps0[i] > 0)
	{

    mdt = min(md[k][i],-small);
    double kk=md[k][i] - dz[k][i]*ed[k][i]*hmn[k][i];
    double tt=kk/mdt;
    hdd[k][i]=tt;
   // printf("=%d %d  %f\n",i+1,k+1,hdd[k][i]);
	}
}

if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
if(eps0[i] > 0)
	{
  qds[k][i] = qsthat[k][i] +(hdd[k][i]-hsthat[k][i])/ rl/100;
    //cout<<i+1<<" "<<k+1<<" "<<qds[k][i]<<" "<<endl;
     //printf("=%d %d  %f\n",i+1,k+1,qds[k][i]);
   // printf("=%d %d  %f %f %f %f %f\n",i+1,k+1,qds[k][i],qsthat[k][i],rl,hdd[k][i],hsthat[k][i]);
	}
}

if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
if(eps0[i] > 0)
	{
	su[k][i] =dz[k][i]/mu[k][i]* (eu[k][i]-du[k][i])*s[k][i];
  qu[k][i] = dz[k][i]/mu[k][i]* (eu[k][i]*q[k][i]- du[k][i]*qst[k][i]);
	}
}

if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
if(eps0[i] > 0)
	{
	cu[k][i] = ((mu[k][i]*su[k][i])- (eu[k][i]-du[k][i])*s[k][i])/(rl/cp);
	}
}

if((k>=1 and k<pver) and (i>=0 and i<ncol))
{
        ql1 = 1/mu[k][i]* (mu[k][i]-dz[k][i]*du[k][i]+dz[k][i]*cu[k][i]);
        ql[k][i] = ql1/ (dz[k][i]*c0mask[i]);          
        totpcp[i] = totpcp[i] + dz[k][i]*(cu[k][i]-du[k][i]);
        rprd[k][i] = c0mask[i]*mu[k][i]*ql[k][i];
}

if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
  qd[k][i] = qds[k][i];
}

double evpp[pver][pcols];//[pver][pcols];
if((k>=0 and k<pver-1) and (i>=0 and i<ncol))
{
if(eps0[i] > 0)
	{
	
    evpp[k][i] = -ed[k][i]*q[k][i] + (md[k][i]*qd[k][i]-md[k+1][i]*qd[k+1][i])/dz[k][i];
    evp[k][i] = max(evpp[k][i],0.0);
   
	}
}

if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
sd[k][i]=s[k][i];
mc[k][i] = mu[k][i] + md[k][i];
cmeg[k][i] = cu[k][i] - evp[k][i];
rprdg[k][i] = rprd[k][i]-evp[k][i];
qlg[k][i]=ql[k][i];
qs[k][i]=qst[k][i];
cug[k][i]=cu[k][i];
evpg[k][i]=evp[k][i];
//printf("=%d %d  %f  %f %f %f\n",i+1,k+1, cmeg[k][i],rprdg[k][i],qs[k][i],cug[k][i]);
}

}


__global__ void zm_convr3()
{
  int i,k;
//double totpcp[pcols];//[pcols]
//double totevp[pcols];//[pcols]
k = blockDim.x * blockIdx.x + threadIdx.x;
i = blockDim.y * blockIdx.y + threadIdx.y;
double zftg=-1;

if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
    du   [k][i] = du   [k][i]* zftg/dp[k][i];
    eu   [k][i] = eu   [k][i]* zftg/dp[k][i];
    ed   [k][i] = ed   [k][i]* zftg/dp[k][i];
    cug  [k][i] = cug  [k][i]* zftg/dp[k][i];
    cmeg [k][i] = cmeg [k][i]* zftg/dp[k][i];
    rprdg[k][i] = rprdg[k][i]* zftg/dp[k][i];
    evpg [k][i] = evpg [k][i]* zftg/dp[k][i];
    //printf("=%d %d  %f  %f %f %f\n",i+1,k+1, du[k][i],cmeg[k][i],rprdg[k][i],evpg[k][i]);
}

}

__device__ double dtpdt[pver][pcols];
__device__ double dqsdtp[pver][pcols];
__device__ double dtmdt[pver][pcols];
__device__ double dqmdt[pver][pcols];
__device__ double dboydt[pver][pcols];
__device__ double thetavp[pver][pcols];
__device__ double thetavm[pver][pcols];
__device__ double dtbdt[pcols];
__device__ double dqbdt[pcols];
__device__ double dtldt[pcols];


__global__ void closure()
{

int k,i;
k = blockDim.x * blockIdx.x + threadIdx.x;
i = blockDim.y * blockIdx.y + threadIdx.y;

double rl = 15.0; 
int il2g=ncol;
double rd=1.4;
double cp=10.0;
double beta=0;
double dadt[pcols];
double debdt;
double dltaa;
double eb;
double grav=9.8;
int il1g=0;
int  kmin, kmax;
double eps1 = 1.0e-8;

if(i>=0 and i<ncol)
{  
        //mb[i] = 0.1;
        eb = p[1][i]*q[1][i]/ (eps1+q[1][i]);
        dtbdt[i] = (1.0/dsubcld[i])* (mu[1][i]*(shat[1][i]-su[1][i])+md[1][i]*(shat[1][i]-sd[1][i]));
        dqbdt[i] = (1.0/dsubcld[i])* (mu[1][i]*(qhat[1][i]-qu[1][i])+md[1][i]* (qhat[1][i]-qd[1][i]));
        debdt = eps1*p[1][i]/ pow((eps1+q[1][i]),2)*dqbdt[i];
        dtldt[i] = -2840.0* (3.5/t[1][i]*dtbdt[i]-debdt/eb)/pow((3.5*log(t[1][i])-log(eb)-4.805),2);
        //printf("clo=%d   %f  %f %f %f\n",i+1,dtbdt[i],dqbdt[i],dtldt[i] );
        //printf("clo=%d   %f  %f %f %f\n",i+1,dqbdt[i],qhat[1][i],qu[1][i],qd[1][i]) ;
}

if((k>=0 and k<pver-1) and (i>=0 and i<ncol))
{
  if (k == jt[i])
  {
    
        dtmdt[k][i] = (1.0/dp[k][i])*(mu[k+1][i]* (su[k+1][i]-shat[k+1][i]- 
                        rl/cp*ql[k+1][i])+md[k+1][i]* (sd[k+1][i]-shat[k+1][i]));

        dqmdt[k][i] = (1.0/dp[k][i])*(mu[k+1][i]* (qu[k+1][i]- 
                        qhat[k+1][i]+ql[k+1][i])+md[k+1][i]*(qd[k+1][i]-qhat[k+1][i]));

   }     //printf("clo=%d  %d %f  %f\n",i+1,k+1,dtmdt[k][i] ,dqmdt[k][i] );  
}

if((k>=0 and k<pver-1) and (i>=0 and i<ncol))
{
  if (k > jt[i])
  {
    
  
        dtmdt[k][i] = (mc[k][i]* (shat[k][i]-s[k][i])+mc[k+1][i]* (s[k][i]-shat[k+1][i]))/ 
                    dp[k][i] - rl/cp*du[k][i]*(beta*ql[k][i]+ (1-beta)*ql[k+1][i]);

        dqmdt[k][i] = (mu[k+1][i]* (qu[k+1][i]-qhat[k+1][i]+cp/rl* (su[k+1][i]-s[k][i]))- 
                    mu[k][i]* (qu[k][i]-qhat[k][i]+cp/rl*(su[k][i]-s[k][i]))+md[k+1][i]* 
                    (qd[k+1][i]-qhat[k+1][i]+cp/rl*(sd[k+1][i]-s[k][i]))-md[k][i]* 
                    (qd[k][i]-qhat[k][i]+cp/rl*(sd[k][i]-s[k][i])))/dp[k][i] + 
                    du[k][i]* (beta*ql[k][i]+(1-beta)*ql[k+1][i]);   

   }     //printf("clo=%d  %d %f  %f\n",i+1,k+1,dtmdt[k][i] ,dqmdt[k][i] );  
}

if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
    thetavp[k][i] = tp[k][i]*pow((1000.0/p[k][i]), (rd/cp)) *(1.0+1.6080*qstp[k][i]-q[mx[i]][i]);
    thetavm[k][i] = t[k][i]* pow((1000.0/p[k][i]), (rd/cp))*(1.0+0.6080*q[k][i]);
    dqsdtp[k][i] = qstp[k][i]* (1.0+qstp[k][i]/eps1)*eps1*rl/(rd*tp[k][i]*tp[k][i]);
    dtpdt[k][i] = tp[k][i]/ (1.0+rl/cp* (dqsdtp[k][i]-qstp[k][i]/tp[k][i]))* 
                        (dtbdt[i]/t[mx[i]][i]+rl/cp* (dqbdt[i]/tl[i]-q[mx[i]][i]/ 
                                pow(tl[i],2)*dtldt[i]));
    dboydt[k][i] = ((dtpdt[k][i]/tp[k][i]+1.0/(1.0+1.6080*qstp[k][i]-q[mx[i]][i])* 
                        (1.6080 * dqsdtp[k][i] * dtpdt[k][i] -dqbdt[i])) - (dtmdt[k][i]/t[k][i]+0.6080/ 
                        (1.0+0.6080*q[k][i])*dqmdt[k][i]))*grav*thetavp[k][i]/thetavm[k][i];  
    //printf("clo=%d  %d %f %f %f %f %f \n",i+1,k+1,thetavp[k][i], thetavm[k][i],dqsdtp[k][i],dtpdt[k][i],dboydt[k][i] );  
        
}

}


__global__ void zm_convr4()
{
int k,i;
k = blockDim.x * blockIdx.x + threadIdx.x;
i = blockDim.y * blockIdx.y + threadIdx.y;



// if (i>=0 and i<ncol)
// {
//  // printf("clo=%d %f  %f \n",i+1,mb[i] ,mumax[i]);        
// }

double grav=9.8;
if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
    mu   [k][i]  = mu   [k][i]*mb[i];
    md   [k][i]  = md   [k][i]*mb[i];
    mc   [k][i]  = mc   [k][i]*mb[i];
    du   [k][i]  = du   [k][i]*mb[i];
    eu   [k][i]  = eu   [k][i]*mb[i];
    ed   [k][i]  = ed   [k][i]*mb[i];
    cmeg [k][i]  = cmeg [k][i]*mb[i];
    rprdg[k][i]  = rprdg[k][i]*mb[i];
    cug  [k][i]  = cug  [k][i]*mb[i];
    evpg [k][i]  = evpg [k][i]*mb[i];   
  // printf("c=%d  %d %f %f %f %f %f %f %f %f %f\n",i+1,k+1,mu[k][i], md[k][i],mc[k][i],du[k][i],eu[k][i] ,ed[k][i],cmeg [k][i],rprdg[k][i],cug[k][i]);   
}

if((k>=0 and k<pver+1) and (i>=0 and i<ncol))
{
    pflxg[k][i]= pflxg[k][i]*mb[i]*100/grav;     
}

}

__global__ void q1q2_pjr()
{
int kbm;
int ktm;
double cp=10.0;
int i,n=1;
int k;
int il1g=0;
int il2g=ncol;
double emc;
double rl=15.0;
double dl[pver][pcols];

k = blockDim.x * blockIdx.x + threadIdx.x;
i = blockDim.y * blockIdx.y + threadIdx.y;

if((k>=0 and k<pver-1) and (i>=0 and i<ncol))
{
        emc = -cug[k][i]+evpg[k][i]  ;                      
        dsdt[k][i] = -rl/cp*emc 
            + (+mu[k+1][i]* (su[k+1][i]-shat[k+1][i]) 
            -mu[k][i]*   (su[k][i]-shat[k][i]) 
            +md[k+1][i]* (sd[k+1][i]-shat[k+1][i]) 
            -md[k][i]*   (sd[k][i]-shat[k][i]) 
            )/dp[k][i];

        dqdt[k][i] = emc + 
            (+mu[k+1][i]* (qu[k+1][i]-qhat[k+1][i]) 
            -mu[k][i]*   (qu[k][i]-qhat[k][i]) 
            +md[k+1][i]* (qd[k+1][i]-qhat[k+1][i]) 
            -md[k][i]*   (qd[k][i]-qhat[k][i]) 
            )/dp[k][i];

        dl[k][i] = du[k][i]*ql[k+1][i];
        dlg[k][i]=dl[k][i];
        //printf("clo=%d  %d %f %f %f \n",i+1,k+1,dsdt[k][i],dqdt[k][i],dlg[k][i] );  
}

}


__global__ void zm_convr5()
{

int k,i;
k = blockDim.x * blockIdx.x + threadIdx.x;
i = blockDim.y * blockIdx.y + threadIdx.y;

int delt=1.0;
double cpres = 10.0;
double grav=9.8;

if((k>=0 and k<pver) and (i>=0 and i<ncol))
{
        q[k][ideep[i]] = (k+1)*(ideep[i]+1) + 2.0*delt*dqdt[k][i];
        qtnd[k][ideep[i]] = dqdt [k][i];
        cme [k][ideep[i]] = cmeg [k][i];
        rprd[k][ideep[i]] = rprdg[k][i];
        zdu [k][ideep[i]] = du   [k][i];
        mcon[k][ideep[i]] = mc   [k][i];
        heat[k][ideep[i]] = dsdt [k][i]*cpres;
        dlf [k][ideep[i]] = dlg  [k][i];
        pflx[k][ideep[i]] = pflxg[k][i];
        ql  [k][ideep[i]] = qlg  [k][i];
        //printf("c=%d  %d %f %f %f %f %f %f %f %f %f %f\n",i+1,k+1,q[k][ideep[i]], qtnd[k][ideep[i]],cme[k][ideep[i]],rprd[k][ideep[i]],zdu[k][ideep[i]] ,mcon[k][ideep[i]],heat[k][ideep[i]],dlf[k][ideep[i]],pflx[k][ideep[i]],ql[k][ideep[i]]);     
}  

if (i>=0 and i<ncol)
{
    jctop[ideep[i]] = jt[i];
    jcbot[ideep[i]] = maxg[i];
    pflx[pver][ideep[i]] = pflxg[pver][i];  
}

if (i>=0 and i<ncol)
{
    prec[i] = grav*max(prec[i],0.0)/ (2.0*delt)/1000.0;   
}

if((k>=0 and k<pver+1) and (i>=0 and i<ncol))
{
     rliq[i] = rliq[i] + dlf[k][i]*dpp[k][i]/grav;  
}  
if (i>=0 and i<ncol)
{
    rliq[i] /=1000;    
}

}

void convr()
{
  int i,k;
double mu_c[pver][pcols];
double dp_c[pver][pcols];
double mumax[pcols];
double mbb[pcols];

cudaMemcpyFromSymbol(mu_c, mu, pver*ncol*sizeof(double));
cudaMemcpyFromSymbol(dp_c,dp , pver*ncol*sizeof(double)); 
for(i=0;i<ncol;i++)
    {  
     mumax[i] = 0;
     mbb[i]=0.1;
    }

  for(k=msg+1;k<pver;k++)
    {  
    for(i=0;i<ncol;i++)
    {   
    mumax[i] = max(mumax[i], mu_c[k][i]/dp_c[k][i]);
    //cout<<i+1<<" "<<k+1<<" "<<mu[k][i]/dp[k][i] <<endl;
    }
    } 

int delt=1.0;

for(i=0;i<ncol;i++)
    {  
     if(mumax[i]>0)
    {
     mbb[i] = min(mbb[i],0.5/(delt*mumax[i]))	;
	 }
	 else
	 {
	 mbb[i] = 0;	
	 }
//	cout<<i+1<<" "<<mbb[i]<<" "<<mumax[i]<<endl;
    }
    cudaMemcpyToSymbol(mu, mu_c, pverp*ncol*sizeof(double));
    cudaMemcpyToSymbol(dp, dp_c, pverp*ncol*sizeof(double));
    cudaMemcpyToSymbol(mb, mbb, ncol*sizeof(double));
}

void zm_convr22()
{
int indexx_c[pcols];
double capelmt =0.0;
double cape_c[pcols];
cudaMemcpyFromSymbol(cape_c, cape, ncol*sizeof(double));
int length=0;

for(int m=0;m<ncol;m++)
{
if (cape_c[m] > capelmt)
	{
        
    indexx_c[length] = m; 
		length = length + 1; 
    //printf("%d %d",i,length); 	    
  }
}

cudaMemcpyToSymbol(indexx, indexx_c, ncol*sizeof(int));
cudaMemcpyToSymbol(lengath, &length, sizeof(int));

}

int main()
{
double t_c[pver][pcols];    
double qh_c[pcnst][pver][pcols];
double pap_c[pver][pcols];
double paph_c[pverp][pcols];
double dpp_c[pver][pcols]; 
double zm_c[pver][pcols];
double geos_c[pcols];
double zi_c[pverp][pcols];
double pblh_c[pcols];//[pcols];//)
double tpert_c[pcols];//[pcols];//)
double landfrac_c[pcols];//[pcols];//) 
double qtnd_c[pver][pcols];//_c[pver][pcols];//_c[pver][pcols];  )       
double heat_c[pver][pcols];//_c[pver][pcols];//_c[pver][pcols]; )         
double mcon_c[pverp][pcols];//(pcols,pverp))
double dlf_c[pver][pcols];//_c[pver][pcols];//_c[pver][pcols];  )   
double pflx_c[pverp][pcols];//(pcols,pverp))  
double cme_c[pver][pcols];//_c[pver][pcols];//_c[pver][pcols]; )
double cape_c[pcols];//[pcols];// )      
double zdu_c[pver][pcols];//_c[pver][pcols];//_c[pver][pcols]; )
double rprd_c[pver][pcols];//_c[pver][pcols];//_c[pver][pcols];  )   
double mu_c[pver][pcols];//_c[pver][pcols];//_c[pver][pcols]; )
double eu_c[pver][pcols];//_c[pver][pcols];//_c[pver][pcols]; )
double du_c[pver][pcols];//_c[pver][pcols];//_c[pver][pcols]; )
double md_c[pver][pcols];//_c[pver][pcols];//_c[pver][pcols]; )
double ed_c[pver][pcols];//_c[pver][pcols];//_c[pver][pcols]; )
double dp_c[pver][pcols];//_c[pver][pcols];//_c[pver][pcols]; )       
double dsubcld_c[pcols];//[pcols];// )     
double jctop_c[pcols];//[pcols];//)  
double jcbot_c[pcols];//[pcols];//)  
double prec_c[pcols];//[pcols];//)
double rliq_c[pcols];//[pcols];// )
double jt_c[pcols];//[pcols];//   )                    
double maxg_c[pcols];//[pcols];//  )                     
int ideep_c[pcols];//[pcols];// )                     
double ql_c[pver][pcols];//_c[pver][pcols];//_c[pver][pcols]; )                  
double zs_c[pcols];//[pcols];//
double dlg_c[pver][pcols];//_c[pver][pcols];//_c[pver][pcols];    
double pflxg_c[pverp][pcols];//(pcols,pverp) 
double cug_c[pver][pcols];//_c[pver][pcols];//_c[pver][pcols];    
double evpg_c[pver][pcols];//_c[pver][pcols];//_c[pver][pcols];    
double mumax_c[pcols];//[pcols];//                   
double pblt_c[pcols];//[pcols];// 
//--------------------------
double q_c[pver][pcols];//_c[pver][pcols];               ! w  grid slice of mixing ratio.
double p_c[pver][pcols];//_c[pver][pcols];               ! w  grid slice of ambient mid-layer pressure in mbs.
double z_c[pver][pcols];//_c[pver][pcols];               ! w  grid slice of ambient mid-layer height in metres.
double s_c[pver][pcols];//_c[pver][pcols];               ! w  grid slice of scaled dry static energy (t+gz/cp).
double tp_c[pver][pcols];//_c[pver][pcols];              ! w  grid slice of parcel temperatures.
double zf_c[pverp][pcols];//(pcols,pver+1)           ! w  grid slice of ambient interface height in metres.
double pf_c[pverp][pcols];//(pcols,pver+1)           ! w  grid slice of ambient interface pressure in mbs.
double qstp_c[pver][pcols];//_c[pver][pcols];            ! w  grid slice of parcel temp. saturation mixing ratio.
double tl_c[pcols];//                  ! w  row of parcel temperature at lcl.
int lcl_c[pcols];//                  ! w  base level index of deep cumulus convection.
int lel_c[pcols];//                  ! w  index of highest theoretical convective plume.
int lon_c[pcols];//                  ! w  index of onset level for deep convection.
int maxi_c[pcols];//                 ! w  index of level with largest moist static energy.
int index_c[pcols];//
double precip;
double qg_c[pver][pcols];//_c[pver][pcols];              ! wg grid slice of gathered values of q.
double tg_c[pver][pcols];//_c[pver][pcols];              ! w  grid slice of temperature at interface.
double pg_c[pver][pcols];//_c[pver][pcols];              ! wg grid slice of gathered values of p.
double zg_c[pver][pcols];//_c[pver][pcols];              ! wg grid slice of gathered values of z.
double sg_c[pver][pcols];//_c[pver][pcols];              ! wg grid slice of gathered values of s.
double tpg_c[pver][pcols];//_c[pver][pcols];             ! wg grid slice of gathered values of tp.
double zfg_c[pverp][pcols];//(pcols,pver+1)          ! wg grid slice of gathered values of zf.
double qstpg_c[pver][pcols];//_c[pver][pcols];           ! wg grid slice of gathered values of qstp.
double ug_c[pver][pcols];//_c[pver][pcols];              ! wg grid slice of gathered values of u.
double vg_c[pver][pcols];//_c[pver][pcols];              ! wg grid slice of gathered values of v.
double cmeg_c[pver][pcols];//_c[pver][pcols]; 
double rprdg_c[pver][pcols];//_c[pver][pcols];            ! wg gathered rain production rate
double capeg_c[pcols];//               ! wg gathered convective available potential energy.
double tlg_c[pcols];//                 ! wg grid slice of gathered values of tl.
double landfracg_c[pcols];//            ! wg grid slice of landfrac
int lclg_c[pcols];//       ! wg gathered values of lcl.
int lelg_c[pcols];//
double dqdt_c[pver][pcols];//_c[pver][pcols];            ! wg mixing ratio tendency at gathered points.
double dsdt_c[pver][pcols];//_c[pver][pcols];            ! wg dry static energy ("temp") tendency at gathered points.
  //  !      real(r8) alpha_c[pver][pcols];//_c[pver][pcols];       ! array of vertical differencing used (=1. for upstream).
double sd_c[pver][pcols];//_c[pver][pcols];              ! wg grid slice of dry static energy in downdraft.
double qd_c[pver][pcols];//_c[pver][pcols];              ! wg grid slice of mixing ratio in downdraft.
double mc_c[pver][pcols];//_c[pver][pcols];              ! wg net upward (scaled by mb) cloud mass flux.
double qhat_c[pver][pcols];//_c[pver][pcols];            ! wg grid slice of upper interface mixing ratio.
double qu_c[pver][pcols];//_c[pver][pcols];              ! wg grid slice of mixing ratio in updraft.
double su_c[pver][pcols];//_c[pver][pcols];              ! wg grid slice of dry static energy in updraft.
double qs_c[pver][pcols];//_c[pver][pcols];              ! wg grid slice of saturation mixing ratio.
double shat_c[pver][pcols];//_c[pver][pcols];            ! wg grid slice of upper interface dry static energy.
double hmn_c[pver][pcols];//_c[pver][pcols];             ! wg moist static energy.
double hsat_c[pver][pcols];//_c[pver][pcols];            ! wg saturated moist static energy.
double qlg_c[pver][pcols];//_c[pver][pcols]; 
double dudt_c[pver][pcols];//_c[pver][pcols];            ! wg u-wind tendency at gathered points.
double dvdt_c[pver][pcols];//_c[pver][pcols];            ! wg v-wind tendency at gathered points.
double mb_c[pcols];//                  ! wg cloud base mass flux.
double  dsubb_c[pcols];

//=======================boyan===========================
int mx_c[pcols];   //=maxi[]
double capeten_c[5][pcols]; //(pcols,5)     //! provisional value of cape
double tv_c[pver][pcols];        
double tpv_c[pver][pcols];       
double buoy_c[pver][pcols]; 

double a1_c[pcols]; 
double a2_c[pcols]; 
double estp_c[pcols]; 
double pl_c[pcols]; 
double plexp_c[pcols]; 
double hmax_c[pcols]; 
int lengath_c=0;
double y_c[pcols]; 

int knt_c[pcols]; 
int lelten_c[5][pcols];
int indexx_c[pcols]; 
//=======================cldprp===========================
double qst_c[pver][pcols];   
double f_c[pver][pcols];//[pver][pcols];
double hu_c[pver][pcols];//[pver][pcols];              ! wg grid slice of dry static energy in updraft.

double dur;
    clock_t start,end;
    
    
	int i,j,k,ii,jj,kk;
	for(i=0;i<pcnst;i++)
    {
    for(j=0;j<pver;j++)
    {
    for(k=0;k<pcols;k++)
    {
    qh_c[i][j][k]=(i+1)*(j+1)*(k+1);
    //cout<<i<<" "<<j<<" "<<k<<" "<<qh_c[i][j][k]<<endl;
    }    
    }
    }
    
    for (ii=0;ii<pver+1;ii++)
    {
    for(jj=0;jj<pcols;jj++)
    {
    paph_c[ii][jj]=(ii+1)*(jj+1)*3;
    zi_c[ii][jj]=(ii+1)+(jj+1)*10;
    }
    }
    
    for (ii=0;ii<pver;ii++)
    {
    for(jj=0;jj<pcols;jj++)
    {
    t_c[ii][jj]=(ii+1)+(jj+1);
    pap_c[ii][jj]=(ii+1)*(jj+1);
    //t_c[ii][jj]=(ii+1)+(jj+1);
    dpp_c[ii][jj]=(ii+1)+(jj+1);
    zm_c[ii][jj]=(ii+1)+(jj+1);
    mu_c[ii][jj]=(ii+1)+(jj+1);
    eu_c[ii][jj]=(ii+1)+(jj+1);
    du_c[ii][jj]=(ii+1)+(jj+1);
    md_c[ii][jj]=(ii+1)+(jj+1);
    ed_c[ii][jj]=(ii+1)+(jj+1);
    dp_c[ii][jj]=(ii+1)+(jj+1);
    }
    }
    
    for (ii=0;ii<pcols;ii++)
	{
    jt_c[ii]=(ii+1)*(ii+1)*7;
    ideep_c[ii]=(ii+1)*(ii+1)*9;
    cape_c[ii]=(ii+1)*(ii+1)*10;
    pblh_c[ii]=(ii+1)*(ii+1)*11;
    tpert_c[ii]=(ii+1)*(ii+1)*12;
    landfrac_c[ii]=(ii+1)*(ii+1)*13;
    geos_c[ii]=(ii+1)*(ii+1)	;
	} 
    for (ii=0;ii<pcols;ii++)
	{
    dsubb_c[ii]=0;
	} 

     
    dim3 ThreadPerBlock(TPBL,TPBN);
    dim3 numBlocks((pver+TPBL-1)/ThreadPerBlock.x,(ncol+TPBN-1)/ThreadPerBlock.y);
    //dim3 ThreadPerBlock2(TPBL,TPBN,TPBG);
   // dim3 numBlocks2((nlayers+TPBL-1)/ThreadPerBlock2.x,(ncol+TPBN-1)/ThreadPerBlock2.y,(ngptlw+TPBG-1)/ThreadPerBlock2.z);
/* 
   cudaMemcpyToSymbol(lengath, &lengath_c, sizeof(int));
   cudaMemcpyToSymbol(paph, paph_c, pverp*ncol*sizeof(double));
   cudaMemcpyToSymbol(zi, zi_c, pverp*ncol*sizeof(double));
   cudaMemcpyToSymbol(t, t_c, pver*ncol*sizeof(double));
   cudaMemcpyToSymbol(pap, pap_c, pver*ncol*sizeof(double));
   cudaMemcpyToSymbol(dpp, dpp_c, pver*ncol*sizeof(double));
   cudaMemcpyToSymbol(zm, zm_c, pver*ncol*sizeof(double));
   cudaMemcpyToSymbol(mu, mu_c, pver*ncol*sizeof(double));
   cudaMemcpyToSymbol(eu, eu_c, pver*ncol*sizeof(double));
   cudaMemcpyToSymbol(du, du_c, pver*ncol*sizeof(double));
   cudaMemcpyToSymbol(md, md_c, pver*ncol*sizeof(double));
   cudaMemcpyToSymbol(ed, ed_c, pver*ncol*sizeof(double));
   cudaMemcpyToSymbol(dp, dp_c, pver*ncol*sizeof(double));
   cudaMemcpyToSymbol(jt, jt_c, ncol*sizeof(double));
   cudaMemcpyToSymbol(ideep, ideep_c, ncol*sizeof(int));
   cudaMemcpyToSymbol(cape, cape_c, ncol*sizeof(double));
   cudaMemcpyToSymbol(pblh, pblh_c, ncol*sizeof(double));
   cudaMemcpyToSymbol(tpert, tpert_c, ncol*sizeof(double));
   cudaMemcpyToSymbol(landfrac, landfrac_c, ncol*sizeof(double));
   cudaMemcpyToSymbol(geos, geos_c, ncol*sizeof(double));
   cudaMemcpyToSymbol(dsubb, dsubb_c, ncol*sizeof(double));
 */
   //cudaMemcpy(lengath_c, lengath,1 * sizeof(int),cudaMemcpyHostToDevice);
   cudaMemcpyToSymbol(lengath, &lengath_c, sizeof(int));
   cudaMemcpy(paph_c, paph, pverp * ncol * sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(zi_c, zi, pverp * ncol * sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(t_c, t, pver * ncol * sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(pap_c, pap, pver * ncol * sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(dpp_c, dpp, pver * ncol * sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(zm_c, zm, pver * ncol * sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(mu_c, mu, pver * ncol * sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(eu_c, eu, pver * ncol * sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(du_c, du, pver * ncol * sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(md_c, md, pver * ncol * sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(ed_c, ed, pver * ncol * sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(dp_c, dp, pver * ncol * sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(jt_c, jt, ncol * sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(ideep_c, ideep, ncol * sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(cape_c, cape, ncol * sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(pblh_c, pblh, ncol * sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(tpert_c, tpert, ncol * sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(landfrac_c, landfrac, ncol * sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(geos_c, geos, ncol * sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(dsubb_c, dsubb, ncol * sizeof(double),cudaMemcpyHostToDevice);  
   //cudaMemcpyToSymbol(, _c, pver*ncol*sizeof(double));
   typedef long clock_t;
    start = clock(); 
    for(int num = 0;num<pcols/ncol;num++){
    zm_convr1<<<numBlocks,ThreadPerBlock>>>();
    buoyan<<<numBlocks,ThreadPerBlock>>>();
    zm_convr22();
    zm_convr2<<<numBlocks,ThreadPerBlock>>>();
    cldprp<<<numBlocks,ThreadPerBlock>>>();
    
    zm_convr3<<<numBlocks,ThreadPerBlock>>>();
    closure<<<numBlocks,ThreadPerBlock>>>();

    convr();

    zm_convr4<<<numBlocks,ThreadPerBlock>>>();  /*(ncols+31)/32,32*/
    q1q2_pjr<<<numBlocks,ThreadPerBlock>>>();
    zm_convr5<<<numBlocks,ThreadPerBlock>>>();
    }
    end = clock();
    dur = (double)(end - start);
   // printf("Use Time:%f ms\n",(dur/CLOCKS_PER_SEC*1000));

    cudaMemcpy(qtnd, qtnd_c, pver * ncol * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(heat, heat_c, pver * ncol * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(mcon, mcon_c, pver * ncol * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(dlf, dlf_c, pver * ncol * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(cme, cme_c, pver * ncol * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(zdu, zdu_c, pver * ncol * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(rprd, rprd_c, pver * ncol * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(mu, mu_c, pver * ncol * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(eu, eu_c, pver * ncol * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(md, md_c, pver * ncol * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(du, du_c, pver * ncol * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(dp, dp_c, pver * ncol * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ed, ed_c, pver * ncol * sizeof(double), cudaMemcpyDeviceToHost);
/*
    cudaMemcpyFromSymbol(qtnd_c, qtnd, pver*ncol*sizeof(double));
    cudaMemcpyFromSymbol(heat_c,heat , pver*ncol*sizeof(double));
    cudaMemcpyFromSymbol(mcon_c, mcon, pver*ncol*sizeof(double));
    cudaMemcpyFromSymbol(dlf_c,dlf , pver*ncol*sizeof(double));
    cudaMemcpyFromSymbol(cme_c, cme, pver*ncol*sizeof(double));
    cudaMemcpyFromSymbol(zdu_c, zdu, pver*ncol*sizeof(double));
    cudaMemcpyFromSymbol(rprd_c,rprd , pver*ncol*sizeof(double));
    cudaMemcpyFromSymbol(mu_c,mu , pver*ncol*sizeof(double));
    cudaMemcpyFromSymbol(eu_c, eu, pver*ncol*sizeof(double));
    cudaMemcpyFromSymbol(md_c,md , pver*ncol*sizeof(double));
    cudaMemcpyFromSymbol(du_c,du , pver*ncol*sizeof(double));
    cudaMemcpyFromSymbol(dp_c, dp, pver*ncol*sizeof(double));
    cudaMemcpyFromSymbol(ed_c, ed, pver*ncol*sizeof(double)); 
*/
    // cudaMemcpyFromSymbol(pblt_c, pblt, ncol*sizeof(double));    
    // cudaMemcpyFromSymbol(p_c,p , pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(pf_c, pf, pverp*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(z_c, z, pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(zf_c,zf , pverp*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(shat_c, shat, pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(qhat_c, qhat, pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(tp_c, tp, pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(qstp_c, qstp, pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(tpv_c, tpv, pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(tv_c, tv, pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(estp_c, estp, pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(qstp_c, qstp, pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(buoy_c, buoy, pver*ncol*sizeof(double));
    
    // cudaMemcpyFromSymbol(qg_c,qg , pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(tg_c,tg , pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(pg_c, pg, pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(zg_c, zg, pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(sg_c, sg, pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(tpg_c, tpg, pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(zfg_c,zfg , pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(qstpg_c,qstpg, pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(ug_c,ug , pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(vg_c, vg, pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(hu_c, hu, pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(su_c, su, pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(qu_c,qu , pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(f_c,f , pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(mc_c,mc , pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(cmeg_c,cmeg , pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(rprdg_c,rprdg , pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(ql_c, ql, pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(md_c,md , pver*ncol*sizeof(double));

    // cudaMemcpyFromSymbol(pflx_c,pflx , pver*ncol*sizeof(double));
    //     
    // //cudaMemcpyFromSymbol(_c, , pver*ncol*sizeof(double));
    // //cudaMemcpyFromSymbol(_c, , pver*ncol*sizeof(double));
    // cudaMemcpyFromSymbol(hmax_c, hmax, ncol*sizeof(double));
    // cudaMemcpyFromSymbol(tl_c, tl, ncol*sizeof(double));
    // cudaMemcpyFromSymbol(plexp_c, plexp, ncol*sizeof(double));
    // cudaMemcpyFromSymbol(pl_c, pl, ncol*sizeof(double));
    // cudaMemcpyFromSymbol(zs_c, zs, ncol*sizeof(double));
    // cudaMemcpyFromSymbol(cape_c, cape, ncol*sizeof(double));
    // cudaMemcpyFromSymbol(indexx_c, indexx, ncol*sizeof(int));
    // cudaMemcpyFromSymbol(ideep_c, ideep, ncol*sizeof(int));
    // cudaMemcpyFromSymbol(dsubcld_c, dsubcld, ncol*sizeof(double));
    // cudaMemcpyFromSymbol(jt_c, jt, ncol*sizeof(double));
    // cudaMemcpyFromSymbol(mb_c, mb, ncol*sizeof(double));
    // cudaMemcpyFromSymbol(&lengath_c, lengath, sizeof(int));

 
   
    // for(k=0;k<pver;k++)
    // {  
    // for(i=0;i<ncol;i++)
    // { 
    // //printf("=%d %d  %f   \n",i+1,k+1, mc_c[k][i]); 
    // //printf("=%d %d  %f   \n",i+1,k+1, du_c[k][i]); 
    // // printf("=%d %d  %f   \n",i+1,k+1, hu_c[k][i]);
    // //printf("=%d %d %f %f %f %f  \n",i+1,k+1, p_c[k][i], pf_c[k][i], z_c[k][i], zf_c[k][i]);
    // //printf("=%d %d  %f %f  \n",i+1,k+1, shat_c[k][i], qhat_c[k][i]);
    // //printf("=%d %d  %f %f %f \n",i+1,k+1, tp_c[k][i], qstp_c[k][i], tpv_c[k][i]);
    // //cout<<q[k][i]<<" "<<qhat[k][i]<<endl;
    // //printf("===%d %d %f %E %f %f %f \n",i+1,k+1, tv_c[k][i], qstp_c[k][i], tp_c[k][i], tpv_c[k][i],buoy_c[k][i]);
    // //printf("=%d %d %f %f %f %f  \n",i+1,k+1, dp_c[k][i], qg_c[k][i], tg_c[k][i], pg_c[k][i]);
    // //printf("=%d %d  %f %f  \n",i+1,k+1, hu_c[k][i],su_c[k][i]);
    // //printf("=%d %d  %f %f  \n",i+1,k+1, mu_c[k][i],eu_c[k][i]);
    // //printf("=%d %d  %f %f %f %f  \n",i+1,k+1, mu_c[k][i],eu_c[k][i],du_c[k][i],hu_c[k][i]);
    // // printf("=%d %d  %f   \n",i+1,k+1, f_c[k][i]);
    // // printf("=%d %d  %f  %f \n",i+1,k+1, su_c[k][i], qu_c[k][i]);
    // }
    // }

    printf("qtnd=  %f   \n",qtnd_c[21][32]); 
    printf("heat=  %f   \n",heat_c[22][43]); 
    printf("mcon=  %f   \n",mcon_c[2][62]); 
    printf("dlf=  %f   \n",dlf_c[33][54]); 
    printf("cme=  %f   \n",cme_c[11][33]); 
    printf("zdu=  %f   \n",zdu_c[23][56]); 
    printf("rprd=  %f   \n",rprd_c[12][37]); 
    printf("mu=  %f   \n",mu_c[8][34]); 
    printf("eu=  %f   \n",eu_c[12][12]); 
    printf("du=  %f   \n",du_c[32][12]); 
    printf("md=  %f   \n",md_c[27][61]); 
    printf("ed=  %f   \n",ed_c[17][61]);
    printf("dp=  %f   \n",dp_c[6][51]); 
  
    return 0;
}


