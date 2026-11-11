#!/usr/bin/env python3
"""UCI Istanbul Stock Exchange (ID 247): univariate classical time-series project.

Mean: zero/mean baseline and ARMA(p,q) over p,q=0..3 selected by train BIC.
Volatility: constant, ARCH(1), GARCH(1,1) via Gaussian quasi-ML.
One-step rolling predictions use information only up to t-1.
No external variables, predictors, or neural networks.

Usage:
  python run_ise_project.py --data data_akbilgic.xlsx --out results --pdf
  python run_ise_project.py --data istanbul_stock_exchange.csv --out results
  python run_ise_project.py --download --out results
  python run_ise_project.py --self-test    # deterministic numerical checks only
"""
from __future__ import annotations
import argparse, io, json, warnings, zipfile, hashlib, platform, re
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot

UCI_PAGE='https://archive.ics.uci.edu/dataset/247/istanbul+stock+exchange'
UCI_FILE='https://archive.ics.uci.edu/static/public/247/istanbul+stock+exchange.zip'
TEAL='#103C49'; ORANGE='#ED9A37'; NAVY='#17313B'; GREY='#53626C'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelcolor':NAVY,'axes.titleweight':'bold',
                     'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white',
                     'savefig.bbox':'tight','savefig.dpi':170})


def download_uci():
    from urllib.request import urlopen
    with urlopen(UCI_FILE,timeout=30) as response:
        raw=response.read()
    z=zipfile.ZipFile(io.BytesIO(raw))
    candidates=[f for f in z.namelist() if f.lower().endswith(('.xlsx','.xls','.csv'))]
    if not candidates: raise ValueError('UCI archive contains no supported Excel/CSV file')
    name=candidates[0]
    return z.read(name),name


FULL_COLUMNS = ['date','ise_tl_return','ise_usd_return','SP','DAX','FTSE','NIKKEI','BOVESPA','EU','EM']


def read_raw(source):
    """Strict loader: explicit headers, no dropped rows, no guessed numeric column.

    Accept the original XLSX/ZIP, a canonical ten-column CSV, or the exported
    two-column CSV. Duplicate ISE/ISE.1 headers are recognized in UCI order.
    CSV dates must be ISO YYYY-MM-DD; ambiguous dates are rejected.
    """
    if source == '__download__':
        raw,name=download_uci()
    else:
        name=str(source); raw=Path(source).read_bytes()
    source_hash=hashlib.sha256(raw).hexdigest()
    archive_hash=None
    if name.lower().endswith('.zip'):
        archive_hash=source_hash
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            matches=[n for n in z.namelist() if Path(n).name=='data_akbilgic.xlsx']
            if len(matches)!=1: raise ValueError('Expected exactly one data_akbilgic.xlsx in ZIP')
            raw=z.read(matches[0]);name=matches[0]
        source_hash=hashlib.sha256(raw).hexdigest()
    if name.lower().endswith('.xlsx'):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            a=pd.read_excel(io.BytesIO(raw),header=None)
        expected=['date','ISE','ISE','SP','DAX','FTSE','NIKKEI','BOVESPA','EU','EM']
        if a.shape[1]!=10 or a.iloc[1].astype(str).tolist()!=expected:
            raise ValueError('Original UCI workbook headers/order not recognized')
        if str(a.iloc[0,1]).strip()!='TL BASED' or str(a.iloc[0,2]).strip()!='USD BASED':
            raise ValueError('Cannot verify TL / USD response identity')
        full=a.iloc[2:].copy();full.columns=FULL_COLUMNS
        full['date']=pd.to_datetime(full['date'],errors='raise')
        parser_warnings=[str(w.message) for w in caught]
    elif name.lower().endswith('.csv'):
        full=pd.read_csv(io.BytesIO(raw),float_precision='round_trip')
        def norm(x):return re.sub(r'[^a-z0-9]','',str(x).lower())
        labels=[norm(x) for x in full.columns]
        if len(labels)==2 and labels[0]=='date' and labels[1] in ('iseusdreturn','usdbasedise'):
            full.columns=['date','ise_usd_return']
        elif len(labels)==10 and labels[0]=='date' and labels[1] in ('isetlreturn','tlbasedise','ise') and labels[2] in ('iseusdreturn','usdbasedise','ise1') and labels[3:]==['sp','dax','ftse','nikkei','bovespa','eu','em']:
            full.columns=FULL_COLUMNS
        else:
            raise ValueError('Unrecognized CSV headers; use exported canonical CSV or original XLSX/ZIP')
        date_strings=full['date'].astype(str)
        if not date_strings.str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():
            raise ValueError('CSV dates must be ISO YYYY-MM-DD; ambiguous dates are not guessed')
        full['date']=pd.to_datetime(date_strings,format='%Y-%m-%d',errors='raise')
        parser_warnings=[]
    else:
        raise ValueError('Use the original XLSX/ZIP or a canonical CSV export')
    full=full.reset_index(drop=True)
    for col in full.columns[1:]:full[col]=pd.to_numeric(full[col],errors='raise').astype(float)
    if full.isna().any().any() or not np.isfinite(full.iloc[:,1:].to_numpy()).all():
        raise ValueError('Missing/nonfinite input observations; nothing is silently dropped')
    if len(full)!=536:raise ValueError(f'Expected the complete 536-record dataset; received {len(full)}')
    if full.date.duplicated().any():raise ValueError('Duplicate dates; no automatic deduplication')
    if not full.date.is_monotonic_increasing:raise ValueError('Dates must already be in chronological order')
    if full.date.iloc[0]!=pd.Timestamp('2009-01-05') or full.date.iloc[-1]!=pd.Timestamp('2011-02-22'):
        raise ValueError('Date range does not match the original UCI workbook')
    if not full.date.equals(full.date.dt.normalize()):raise ValueError('Expected dates without intraday times')
    df=full[['date','ise_usd_return']].copy()
    meta={'source_filename':name,'source_sha256':source_hash,'archive_sha256':archive_hash,
          'observations':len(full),'input_columns':full.columns.tolist(),'missing_values':0,'duplicate_dates':0,
          'first_date':str(df.date.iloc[0].date()),'last_date':str(df.date.iloc[-1].date()),
          'original_response_col':'Third workbook column: ISE / USD BASED',
          'external_predictors':0,'new_predictors_engineered':0,'parser_warnings':parser_warnings,
          'uci_metadata_discrepancy':'Web prose says Jun 5, 2009; original workbook starts 2009-01-05. SP contains continuous returns, despite the web Binary label.'}
    return df,meta,full


def safe_test(fun, data, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        v=fun(data,**kwargs)
    pv=float(v[1]);relation='='
    if fun.__name__=='kpss':
        if pv==.1 and float(v[0])<float(v[3]['10%']):relation='>'
        elif pv==.01 and float(v[0])>float(v[3]['1%']):relation='<'
    if pv==0: relation='numerical_underflow'
    return {'statistic':float(v[0]),'p_value':pv,'p_value_relation':relation,
            'used_lags':int(v[2]),'warnings':[str(w.message) for w in caught]}


def ljung(x,lag=10,model_df=0):
    v=acorr_ljungbox(x,lags=[lag],model_df=model_df,return_df=True)
    return {'lags':lag,'model_df':model_df,'reference_df':lag-model_df,
            'Q':float(v['lb_stat'].iloc[0]),'p_value':float(v['lb_pvalue'].iloc[0])}


def qml_recursion(errors,omega,alpha,beta):
    e=np.asarray(errors,dtype=float)
    h=np.empty(len(e));h[0]=max(float(np.mean(e**2)),1e-8)
    for t in range(1,len(e)):
        h[t]=omega+alpha*e[t-1]**2+beta*h[t-1]
    return h


def qml_fit(errors,model):
    """Two-stage conditional Gaussian QML, fixing the previously fitted mean.

    All models use the SAME fixed training backcast h[0]=mean(e**2), and
    score only t=1,...,n-1. BIC counts variance parameters, conditional on
    the common mean fit; it is not a jointly estimated ARMA-GARCH BIC.
    Only successful feasible optimizer results can be selected.
    """
    e=np.asarray(errors,dtype=float);sigma2=float(np.mean(e[1:]**2))
    n_eff=len(e)-1
    if sigma2<=0 or not np.isfinite(e).all():raise ValueError('Invalid QML errors')
    def calc(x):
        beta=x[2] if len(x)==3 else 0.
        h=qml_recursion(e,x[0],x[1],beta)[1:]
        if np.any(h<=0) or not np.isfinite(h).all():return 1e100
        return float(.5*np.sum(np.log(2*np.pi*h)+e[1:]**2/h))
    attempts=[]
    if model=='constant':
        omega=sigma2;alpha=beta=0.;k=1
        h=qml_recursion(e,omega,alpha,beta);loss=calc([omega,alpha])
    else:
        starts=([[sigma2*(1-a),a] for a in [.05,.2,.5,.8]] if model=='ARCH(1)' else
                [[sigma2*(1-a-b),a,b] for a,b in [(.1,.8),(.2,.5),(.1,.2),(.05,.94)]])
        bounds=[(1e-10,None),(0.,.999)] + ([(0.,.999)] if model=='GARCH(1,1)' else [])
        constraint={'type':'ineq','fun':lambda x:.999-x[1]-(x[2] if len(x)>2 else 0)}
        fits=[]
        for start in starts:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                res=minimize(calc,start,bounds=bounds,method='SLSQP',constraints=[constraint],
                             options={'maxiter':1500,'ftol':1e-10})
            feasible=(np.isfinite(res.x).all() and res.x[0]>0 and np.all(res.x[1:]>=0)
                      and sum(res.x[1:])<=.999+1e-8)
            accepted=bool(res.success and feasible and np.isfinite(res.fun))
            attempts.append({'success':bool(res.success),'accepted':accepted,'message':str(res.message),
                             'warnings':[str(w.message) for w in caught]})
            if accepted:fits.append(res)
        if not fits:raise RuntimeError(f'{model}: no converged feasible optimization: {attempts}')
        best=min(fits,key=lambda r:r.fun)
        omega,alpha=map(float,best.x[:2]);beta=float(best.x[2]) if len(best.x)==3 else 0.
        k=len(best.x);loss=calc(best.x);h=qml_recursion(e,omega,alpha,beta)
    return {'name':model,'omega':omega,'alpha':alpha,'beta':beta,'loglik':-loss,
            'aic':2*k+2*loss,'bic':np.log(n_eff)*k+2*loss,'variance':h.tolist(),
            'converged':True,'n_effective':n_eff,'variance_parameter_count':k,
            'initial_variance':float(h[0]),'optimization_attempts':attempts}


def train_arma(ytrain, max_order=3):
    candidates=[]; fits={}
    for p in range(max_order+1):
        for q in range(max_order+1):
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    fit=ARIMA(ytrain,order=(p,0,q),trend='c',enforce_stationarity=True,
                              enforce_invertibility=True).fit(method_kwargs={'maxiter':250})
                converged=bool(fit.mle_retvals.get('converged',True))
                candidates.append({'p':p,'q':q,'aic':float(fit.aic),'bic':float(fit.bic),
                                   'converged':converged,'warnings':' | '.join(str(w.message) for w in caught)})
                if converged and np.isfinite(fit.bic): fits[(p,q)]=fit
            except Exception as exc:
                candidates.append({'p':p,'q':q,'aic':np.nan,'bic':np.nan,'converged':False,'error':str(exc)[:150]})
    if not fits:raise RuntimeError('No ARMA candidate converged')
    best=min(fits,key=lambda order: fits[order].bic)
    return fits[best],best,pd.DataFrame(candidates)


def rolling_arma(fit, test):
    # One step ahead, then update with observed return; no future observations used.
    predictions=[];current=fit
    for val in test:
        p=float(np.asarray(current.forecast(steps=1))[0]);predictions.append(p)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            current=current.append(np.array([val]),refit=False)
    return np.asarray(predictions)


def rmse(actual,pred):return float(np.sqrt(np.mean((np.asarray(actual)-np.asarray(pred))**2)))
def mae(actual,pred):return float(np.mean(np.abs(np.asarray(actual)-np.asarray(pred))))

def variance_forecasts(fitted,train_errors,test_errors):
    e0=np.asarray(train_errors);e1=np.asarray(test_errors)
    htrain=np.asarray(fitted['variance']); hnext=float(htrain[-1]);last=float(e0[-1]); forecasts=[]
    omega,alpha,beta=[fitted[k] for k in ('omega','alpha','beta')]
    if fitted['name']=='constant':return np.full(len(e1),omega)
    for et in e1:
        hnext=omega+alpha*last*last+beta*hnext
        hnext=max(float(hnext),1e-9)
        forecasts.append(hnext);last=float(et)
    return np.asarray(forecasts)


def variance_score(e,predvar):
    h=np.maximum(np.asarray(predvar),1e-8); x=np.asarray(e)**2
    return {'qlike':float(np.mean(np.log(h)+x/h)),
            'variance_proxy_mse':float(np.mean((x-h)**2))}


def make_plot(path,draw):
    fig,ax=plt.subplots(figsize=(10.5,4.6))
    draw(fig,ax)
    fig.savefig(path);plt.close(fig)


def empirical_plots(df, y, ntrain, best, order, grid, meanpred,meanmetrics,variances,varmetrics,
                    residtrain,stdresid, out):
    f=out/'figures';f.mkdir(parents=True,exist_ok=True)
    x=df.date;train_x=x.iloc[:ntrain];test_x=x.iloc[ntrain:]
    def series(fig,ax):
        ax.plot(x,y,lw=.9,color=TEAL);ax.axvline(x.iloc[ntrain],color=ORANGE,ls='--',label='Train / test')
        ax.set(ylabel='Daily return (%)',title='ISE index daily returns (USD-based)',xlabel='Date');ax.legend()
    make_plot(f/'01_returns.png',series)
    def histogram(fig,ax):
        ax.hist(y[:ntrain],bins=28,density=True,alpha=.68,color=TEAL,edgecolor='white')
        xx=np.linspace(np.min(y[:ntrain]),np.max(y[:ntrain]),150)
        ax.plot(xx,stats.norm.pdf(xx,np.mean(y[:ntrain]),np.std(y[:ntrain])),color=ORANGE,lw=2,label='Normal fit')
        ax.set(title='Training returns: distribution versus Gaussian',xlabel='Daily return (%)');ax.legend()
    make_plot(f/'02_distribution.png',histogram)
    for filename,arr,title in [('03_acf.png',y[:ntrain],'Return ACF'),('04_squared_acf.png',(y[:ntrain]-np.mean(y[:ntrain]))**2,'Squared centered-return ACF'),
                               ('10_residual_acf.png',residtrain,'ARMA residual ACF'),('11_residual_squared_acf.png',residtrain**2,'ARMA residual-squared ACF')]:
        def draw(fig,ax,series=arr,ttl=title):
            plot_acf(series, lags=min(30,len(series)//4),ax=ax,zero=False,color=TEAL)
            ax.set(title=ttl,xlabel='Lag')
        make_plot(f/filename,draw)
    def pacf_plot(fig,ax):
        plot_pacf(y[:ntrain],lags=min(25,len(y[:ntrain])//4),ax=ax,zero=False,method='ywm')
        ax.set(title='Return PACF',xlabel='Lag')
    make_plot(f/'05_pacf.png',pacf_plot)
    def roll(fig,ax):
        # Rolling SD is purely descriptive; never given to an ARMA/GARCH model as an external predictor.
        rollstd=pd.Series(y).rolling(20).std()
        ax.plot(x,rollstd,color=TEAL,lw=1.4)
        ax.axvline(x.iloc[ntrain],color=ORANGE,ls='--')
        ax.set(title='20-observation moving standard deviation (descriptive only)',ylabel='SD (%)',xlabel='Date')
    make_plot(f/'06_rolling_sd.png',roll)
    def aicbic(fig,ax):
        g=grid[grid.converged].sort_values('bic');labels=[f'({int(a)},{int(b)})' for a,b in zip(g.p,g.q)]
        pos=np.arange(len(g));ax.bar(pos-.19,g.aic,width=.38,label='AIC',color=TEAL)
        ax.bar(pos+.19,g.bic,width=.38,label='BIC',color=ORANGE)
        ax.set_xticks(pos,labels,rotation=40);ax.set(title='ARMA order selection on training data',xlabel='(p, q)',ylabel='Information criterion');ax.legend()
    make_plot(f/'07_aic_bic.png',aicbic)
    def forecast(fig,ax):
        actual=y[ntrain:];p=meanpred['ARMA']
        ax.plot(test_x,actual,lw=1.05,color=TEAL,label='Observed return')
        ax.plot(test_x,p,lw=1.4,color=ORANGE,label=f'ARMA{order} rolling one-step forecast')
        ax.set(title='Out-of-sample one-step return forecasts',ylabel='Daily return (%)',xlabel='Date');ax.legend()
    make_plot(f/'08_forecast_returns.png',forecast)
    def errors(fig,ax):
        err=y[ntrain:]-meanpred['ARMA'];ax.plot(test_x,err,color=TEAL,lw=1)
        ax.axhline(0,ls='--',color=ORANGE);ax.set(title='Holdout forecast errors',ylabel='Percentage points',xlabel='Date')
    make_plot(f/'09_forecast_errors.png',errors)
    def qq(fig,ax):
        qqplot(residtrain,line='s',ax=ax)
        ax.set(title='ARMA training residual normal Q-Q plot')
    make_plot(f/'12_residual_qq.png',qq)
    def vol(fig,ax):
        ax.plot(test_x,np.abs(y[ntrain:]-meanpred['ARMA']),color='#A0ABB3',lw=1,alpha=.8,label='|forecast error|')
        for name,h in variances.items():
            ax.plot(test_x,np.sqrt(h),lw=1.4,label=name)
        ax.set(title='One-step conditional standard deviation forecasts',ylabel='Percentage points',xlabel='Date');ax.legend(ncol=2,fontsize=8)
    make_plot(f/'13_volatility_forecast.png',vol)
    def normalized(fig,ax):
        ax.plot(train_x.iloc[-len(stdresid):],stdresid,color=TEAL,lw=1)
        ax.axhline(0,color=ORANGE,ls='--')
        ax.set(title='Training standardized residuals: ARMA + GARCH',ylabel='Residual / conditional SD',xlabel='Date')
    make_plot(f/'14_standardized_residual.png',normalized)
    def meancompare(fig,ax):
        labels=list(meanmetrics.keys());a=[meanmetrics[k]['rmse'] for k in labels]
        ax.barh(labels,a,color=[ORANGE if v=='ARMA' else TEAL for v in labels])
        ax.invert_yaxis();ax.set(title='Holdout mean forecasts (lower RMSE is better)',xlabel='RMSE, percentage points')
    make_plot(f/'15_mean_comparison.png',meancompare)
    def varcompare(fig,ax):
        labels=list(varmetrics.keys());a=[varmetrics[k]['qlike'] for k in labels]
        ax.barh(labels,a,color=[ORANGE if 'GARCH' in v else TEAL for v in labels])
        ax.invert_yaxis();ax.set(title='Holdout variance forecasts: QLIKE (lower is better)',xlabel='Average log(h) + residual²/h')
    make_plot(f/'16_variance_comparison.png',varcompare)
    for filename,arr,title in [('17_garch_std_acf.png',stdresid[1:],'GARCH standardized-residual ACF'),
                               ('18_garch_std_squared_acf.png',stdresid[1:]**2,'GARCH squared-standardized-residual ACF')]:
        def draw(fig,ax,series=arr,ttl=title):
            plot_acf(series,lags=30,ax=ax,zero=False)
            ax.set(title=ttl,xlabel='Lag (observed Istanbul trading sessions)')
        make_plot(f/filename,draw)
    def standard_qq(fig,ax):
        qqplot(stdresid[1:],line='s',ax=ax)
        ax.set(title='GARCH standardized-residual normal Q-Q')
    make_plot(f/'19_garch_std_qq.png',standard_qq)
    return sorted(f.glob('*.png'))


def run(source,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    df,source_meta,full=read_raw(source)
    full.to_csv(out/'ISE_full_original.csv',index=False,date_format='%Y-%m-%d')
    df.to_csv(out/'used_original_univariate_series.csv',index=False)
    y=100*df.ise_usd_return.to_numpy(float)   # Percent unit scaling, not an additional predictor.
    ntrain=int(len(y)*.8)
    train=y[:ntrain];test=y[ntrain:]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tests={'adf':safe_test(adfuller,train,autolag='AIC'),
               'kpss':safe_test(kpss,train,regression='c',nlags='auto')}
    fit,order,grid=train_arma(train)
    grid.to_csv(out/'arma_order_comparison.csv',index=False)
    arma_pred=rolling_arma(fit,test)
    mean_pred={'Zero return':np.zeros(len(test)),'Historical mean':np.full(len(test),train.mean()),'ARMA':arma_pred}
    mean_metrics={k:{'rmse':rmse(test,v),'mae':mae(test,v)} for k,v in mean_pred.items()}
    pd.DataFrame(mean_metrics).T.to_csv(out/'mean_forecast_comparison.csv')
    residual_train=np.asarray(fit.resid,dtype=float)
    # ARIMA is stationary-initialized here, not diffuse. No arbitrary 10-day discard.
    # The variance likelihood consistently conditions on its first backcast.
    eres=residual_train
    lm,lm_p,_,_=het_arch(eres,nlags=5,ddof=sum(order))
    diagnostics={'returns_ljung_box':ljung(train),
                 'returns_squared_ljung_box':ljung((train-np.mean(train))**2),
                 'arma_residual_ljung_box':ljung(eres,model_df=sum(order)),
                 'arma_residual_squared_ljung_box':ljung(eres**2),
                 'arch_lm_5':{'p_value':float(lm_p),'statistic':float(lm)},
                 'jarque_bera_mean_residuals':{'statistic':float(stats.jarque_bera(eres).statistic),'p_value':float(stats.jarque_bera(eres).pvalue)}}
    volatility_models={name:qml_fit(eres,name) for name in ['constant','ARCH(1)','GARCH(1,1)']}
    selected_variance=min(volatility_models,key=lambda k:volatility_models[k]['bic'])
    etest=test-arma_pred
    variance_preds={name:variance_forecasts(v,eres,etest) for name,v in volatility_models.items()}
    varmetrics={name:variance_score(etest,v) for name,v in variance_preds.items()}
    pd.DataFrame(varmetrics).T.to_csv(out/'variance_forecast_comparison.csv')
    pd.DataFrame([{k:val for k,val in v.items() if k!='variance'} for v in volatility_models.values()]).to_csv(out/'variance_estimates.csv',index=False)
    # The GARCH standardized residual diagnostics use training errors / QML variance.
    garch=volatility_models['GARCH(1,1)']
    standard=eres/np.sqrt(np.asarray(garch['variance']))
    diagnostics['garch_stdresid_ljung_box']=ljung(standard[1:],model_df=sum(order))
    diagnostics['garch_stdresid_squared_ljung_box']=ljung(standard[1:]**2)
    diagnostics['p_value_caveat']='Reference p-values are approximate after fitting/selection; ordinary Ljung-Box on raw ARMA residuals is not heteroskedasticity-robust. GARCH squared-standardized-residual p-values are descriptive, not a formal adjusted GARCH portmanteau test.'
    summary=pd.DataFrame({label:{'n':len(z),'mean':np.mean(z),'sd':np.std(z,ddof=1),'min':np.min(z),'max':np.max(z),'skewness':stats.skew(z,bias=False),'excess_kurtosis':stats.kurtosis(z,bias=False)} for label,z in [('training',train),('test_descriptive_only',test)]}).T
    summary.to_csv(out/'summary_statistics.csv')
    source_meta.update({'response_unit':'100 x original USD-based ISE daily return (percent units)',
                        'train_n':int(ntrain),'test_n':int(len(test)),'train_end':str(df.date.iloc[ntrain-1].date()),
                        'test_start':str(df.date.iloc[ntrain].date())})
    result={'source':source_meta,'test_results':tests,'diagnostics':diagnostics,
            'arma_order':list(order),'arma_aic':float(fit.aic),'arma_bic':float(fit.bic),
            'summary_statistics':summary.to_dict(orient='index'),
            'selected_variance_by_training_bic':selected_variance,
            'variance_fit_method':'Two-stage conditional Gaussian QML; fixed common mean; common first density omitted; all 428 training residuals retained',
            'package_versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','statsmodels','matplotlib','openpyxl','reportlab']},
            'python_version':platform.python_version(),
            'arma_parameters':{k:float(v) for k,v in zip(fit.param_names,fit.params)},
            'mean_forecast_metrics':mean_metrics,
            'volatility_parameters':{k:{x:v for x,v in val.items() if x!='variance'} for k,val in volatility_models.items()},
            'variance_forecast_metrics':varmetrics,
            'lowest_test_rmse_descriptive':min(mean_metrics,key=lambda k:mean_metrics[k]['rmse']),
            'lowest_test_qlike_descriptive':min(varmetrics,key=lambda k:varmetrics[k]['qlike']),
            'note':'Results from actual observations read from input; no machine learning, no external predictors; variance loss uses squared one-step forecast errors as noisy volatility proxy.'}
    with open(out/'results.json','w') as f:json.dump(result,f,indent=2,allow_nan=False)
    preds=pd.DataFrame({'date':df.date.iloc[ntrain:].values,'observed_return_percent':test,
                         'forecast_zero':mean_pred['Zero return'],'forecast_mean':mean_pred['Historical mean'],
                         'forecast_arma':arma_pred,'forecast_error_arma':etest,
                         **{f'variance_{name}':v for name,v in variance_preds.items()}})
    preds.to_csv(out/'test_forecasts.csv',index=False)
    figs=empirical_plots(df,y,ntrain,fit,order,grid,mean_pred,mean_metrics,variance_preds,varmetrics,
                         eres,standard,out)
    fit_summary=fit.summary().as_text()
    (out/'arma_fit_summary.txt').write_text(fit_summary,encoding='utf-8')
    train_diagnostic=pd.DataFrame({'date':df.date.iloc[:ntrain].values,'return_percent':train,'mean_residual':eres,**{f'variance_{k}':v['variance'] for k,v in volatility_models.items()},'garch_standardized_residual':standard})
    train_diagnostic.to_csv(out/'training_diagnostics.csv',index=False)
    print('UCI original return series:',source_meta)
    print('Best ARMA by training BIC:',order,'train BIC',fit.bic)
    print('Mean test metrics:');print(pd.DataFrame(mean_metrics).T)
    print('Variance test metrics:');print(pd.DataFrame(varmetrics).T)
    print('Generated',len(figs),'empirical plots under',out/'figures')
    return result


def build_pdf(r, out):
    """Generate the full slide-style PDF from this run's numbers and figures."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor, white
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from matplotlib.font_manager import findfont, FontProperties
    from xml.sax.saxutils import escape
    W,H=960,540
    for name,weight,style in [('ReportSans','normal','normal'),('ReportSans-Bold','bold','normal'),
                              ('ReportSans-Italic','normal','oblique'),('ReportSans-BoldItalic','bold','oblique')]:
        fontfile=findfont(FontProperties(family='DejaVu Sans',weight=weight,style=style))
        pdfmetrics.registerFont(TTFont(name,fontfile))
    pdfmetrics.registerFontFamily('ReportSans',normal='ReportSans',bold='ReportSans-Bold',
                                  italic='ReportSans-Italic',boldItalic='ReportSans-BoldItalic')
    destination=out/'ISE_Real_Data_NPC_Style.pdf'
    c=canvas.Canvas(str(destination),pagesize=(W,H))
    c.setTitle('ISE returns: audited classical time-series study')
    c.setAuthor('Ritwik Bhattacharya')
    page=0
    body=ParagraphStyle('body',fontName='ReportSans',fontSize=16,leading=22,textColor=HexColor(NAVY))
    small=ParagraphStyle('small',parent=body,fontSize=13,leading=18)
    cell=ParagraphStyle('cell',parent=body,fontSize=14,leading=19)
    whitecell=ParagraphStyle('whitecell',parent=cell,textColor=white,fontName='ReportSans-Bold')
    def new(title,tag='CLASSICAL TIME-SERIES STUDY | VERIFIED AGAINST ORIGINAL WORKBOOK'):
        nonlocal page
        if page:c.showPage()
        page+=1
        c.setFillColor(HexColor(TEAL));c.rect(0,H-74,W,74,fill=1,stroke=0)
        c.setFillColor(white);c.setFont('ReportSans-Bold',25);c.drawString(40,H-47,title)
        c.setFillColor(HexColor(ORANGE));c.rect(40,H-84,64,4,fill=1,stroke=0)
        c.setFont('ReportSans',9);c.drawString(40,H-103,tag)
        c.setStrokeColor(HexColor('#D5E0E3'));c.line(40,34,W-40,34)
        c.setFillColor(HexColor(GREY));c.setFont('ReportSans',10)
        c.drawString(40,19,'UCI ISE | USD-based return | 428 training + 108 test observations')
        c.drawRightString(W-40,19,str(page))
    def para(text,y=405,size='body',x=48,width=864):
        p=Paragraph(text,small if size=='small' else body)
        _,h=p.wrap(width,H);p.drawOn(c,x,y-h)
        if y-h<44:raise RuntimeError(f'PDF content exceeds page {page}')
        return y-h-14
    def bullets(lines,y=405):
        for line in lines:
            c.setFillColor(HexColor(ORANGE));c.circle(49,y-12,3,fill=1,stroke=0)
            y=para(line,y,x=65,width=845)
        return y
    def equation(expr,y=399,height=52):
        buf=io.BytesIO();fig=plt.figure(figsize=(11,1.05))
        fig.text(.015,.45,'$'+expr+'$',fontsize=22,color=NAVY,va='center')
        fig.savefig(buf,format='png',dpi=220,bbox_inches='tight',pad_inches=.06);plt.close(fig)
        buf.seek(0);im=ImageReader(buf);iw,ih=im.getSize();scale=min(850/iw,height/ih)
        c.drawImage(im,48,y-ih*scale,width=iw*scale,height=ih*scale,mask='auto')
        return y-ih*scale-16
    def table(headers,rows,widths=None,y=400):
        widths=widths or [864/len(headers)]*len(headers)
        vals=[[Paragraph(escape(str(v)),whitecell) for v in headers]]
        vals += [[Paragraph(escape(str(v)),cell) for v in row] for row in rows]
        t=Table(vals,colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor(TEAL)),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor('#F0F5F6'),white]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),12),
            ('RIGHTPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,-1),10),
            ('BOTTOMPADDING',(0,0),(-1,-1),10)]))
        _,h=t.wrap(864,450);t.drawOn(c,48,y-h)
        if y-h<44:raise RuntimeError(f'Table exceeds page {page}')
        return y-h-20
    def plot_page(title,name,note):
        new(title,'EMPIRICAL RESULTS | NUMBERS AND PLOTS REGENERATED BY THE ATTACHED SCRIPT')
        im=ImageReader(str(out/'figures'/name));iw,ih=im.getSize();scale=min(864/iw,325/ih)
        w,h=iw*scale,ih*scale;c.drawImage(im,(W-w)/2,96+(325-h)/2,width=w,height=h)
        para(note,83,'small')
    def pv(d):
        if d.get('p_value_relation')=='numerical_underflow':return '< 1e-16 (numerical zero)'
        return d.get('p_value_relation','=')+' '+f"{d['p_value']:.5g}"
    s=r['source'];d=r['diagnostics'];v=r['volatility_parameters'];mm=r['mean_forecast_metrics'];vm=r['variance_forecast_metrics']
    new('MODELLING STOCK-INDEX RETURNS','AUDITED AND RECOMPUTED | 11 NOVEMBER 2026')
    para('A classical univariate study of ISE 100 USD-based returns',393)
    para('<b>ARMA | ARCH | GARCH</b>',321)
    para('<b>Ritwik Bhattacharya</b><br/>M.Stat Programme | Indian Statistical Institute, Delhi',244)
    para('536 historical observations, 2009-2011. Original values retained; no external predictors or machine-learning methods.',136,'small')
    new('What the audit established')
    bullets(['The original script reproduces the old PDF scores to their displayed precision when run on the supplied UCI workbook.',
             'The data and main numerical findings were valid. Several validation, estimation and interpretation details needed correction.',
             'This PDF replaces every affected table and plot with results from the corrected script.',
             'The final slide separates mean forecasting from volatility forecasting; complexity alone is not a success criterion.'])
    new('Original data and provenance')
    y=table(['Item','Verified from the uploaded workbook'],[
        ['File','data_akbilgic.xlsx inside istanbul+stock+exchange.zip'],
        ['Shape','536 data rows; date + 9 numerical return columns'],
        ['Dates','2009-01-05 to 2011-02-22'],
        ['Integrity','No missing cells, duplicate dates or nonfinite numeric values'],
        ['Ordering','Already chronological; no sorting, deletion or imputation']], [190,674])
    para('The UCI page says "Jun 5, 2009"; actual workbook dates start on 5 January. Its "Binary" label for SP also conflicts with the continuous workbook values. Workbook values govern this analysis. [1]',y,'small')
    new('Response and excluded variables')
    y=equation(r'Y_t=100R_t^{\mathrm{ISE,USD}}')
    bullets(['Response: the third workbook column, labelled ISE under USD BASED. The second column is TL-based ISE.',
             'Multiplication by 100 only changes units. We do not recalculate returns, assume a simple/log-return formula, or create predictors.',
             'Excluded: TL-based ISE, SP, DAX, FTSE, NIKKEI, BOVESPA, EU and EM. Date identifies observation order.',
             'A lag is one successive observed Istanbul trading session. Calendar gaps are not filled.'],y)
    new('Chronological evaluation protocol')
    y=table(['Block','Dates','Observations'],[
        ['Training','2009-01-05 to 2010-09-15','428'],
        ['Test','2010-09-16 to 2011-02-22','108']],[170,520,174])
    bullets(['Choose ARMA order and variance model by training criteria; estimate all parameters using training data only.',
             'Forecast one step, observe that return, then update the state. Parameters remain fixed throughout the test block.',
             'Test rankings describe this one split. The test sample has already been examined in the earlier report; it is not a fresh independent confirmation.'],y)
    new('Training summary: percent-unit returns')
    summ=r['summary_statistics']['training']
    table(['Statistic','Value'],[['Mean',f"{summ['mean']:.6f}"],['Sample SD',f"{summ['sd']:.6f}"],
          ['Minimum / maximum',f"{summ['min']:.6f} / {summ['max']:.6f}"],['Skewness',f"{summ['skewness']:.6f}"],
          ['Excess kurtosis',f"{summ['excess_kurtosis']:.6f}"]],[390,474])
    new('Stationarity: evidence and limits')
    y=equation(r'E(Y_t)=\mu,\quad \mathrm{Cov}(Y_t,Y_{t-h})=\gamma(h),\quad E(Y_t^2)<\infty')
    y=table(['Training test','Statistic','p-value','Lags'],[
        ['ADF, intercept',f"{r['test_results']['adf']['statistic']:.6f}",pv(r['test_results']['adf']),r['test_results']['adf']['used_lags']],
        ['KPSS, level',f"{r['test_results']['kpss']['statistic']:.6f}",pv(r['test_results']['kpss']),r['test_results']['kpss']['used_lags']]], [240,160,370,94],y)
    para('ADF rejects a unit root; KPSS does not reject level stationarity at 5%. These support using d=0 for returns, but do not prove stationarity or exclude structural breaks. ADF uses AIC lag selection; KPSS uses automatic bandwidth. [2]',y,'small')
    new('Mean model: match the code parameterization')
    y=equation(r'Y_t-\mu=\sum_{i=1}^{p}\phi_i(Y_{t-i}-\mu)+\epsilon_t+\sum_{j=1}^{q}\theta_j\epsilon_{t-j}',height=80)
    y=equation(r'E(\epsilon_t\mid\mathcal{F}_{t-1})=0,\qquad c=\mu\left(1-\sum_{i=1}^p\phi_i\right)',y,height=65)
    bullets(['F(t-1) is the history available before observing Y(t). Past returns and innovations are internal model terms.',
             'statsmodels ARIMA(..., trend="c") reports the level mean mu as const. It is not generally the recursion intercept c. [3]',
             'ARMA(0,0) gives a constant mean. Its estimated mean nearly equals the arithmetic training mean.'],y)
    new('Order selection: training BIC')
    y=equation(r'\mathrm{AIC}=-2\ell+2k,\qquad \mathrm{BIC}=-2\ell+k\log(n)')
    bullets(['Fit p,q = 0,...,3, with d=0 and a constant, to the same 428 training returns. Enforce stationarity and invertibility.',
             'Only converged finite fits enter the choice. Here k=p+q+2 counts the mean and variance as well as AR/MA terms.',
             f"Selected order: ARMA{tuple(r['arma_order'])}; AIC={r['arma_aic']:.4f}, BIC={r['arma_bic']:.4f}.",
             'This is a Gaussian working-likelihood order criterion. It does not jointly estimate ARMA-GARCH or establish significant coefficients under heteroskedasticity.'],y)
    new('One-step forecasts and mean loss')
    y=equation(r'\widehat Y_{t|t-1},\quad e_t=Y_t-\widehat Y_{t|t-1},\quad t=429,\ldots,536')
    y=equation(r'\mathrm{RMSE}=\sqrt{\frac{1}{108}\sum e_t^2},\qquad \mathrm{MAE}=\frac{1}{108}\sum |e_t|',y)
    bullets(['Benchmarks: forecast zero, or the fixed training mean. Selected ARMA uses past available observations with fixed fitted parameters.',
             'Use the same 108 test observations for each model. Errors are in percentage points.',
             'RMSE is the primary mean criterion; MAE is supplementary. MAPE is unsuitable for returns near or crossing zero.'],y)
    new('Residual dependence and ARCH effects')
    y=equation(r'\widehat\epsilon_t^2=a_0+\sum_{j=1}^{5}a_j\widehat\epsilon_{t-j}^2+u_t')
    y=equation(r'H_0:a_1=\cdots=a_5=0,\qquad \mathrm{LM}=(n_{\rm aux}-p-q)R^2',y)
    bullets(['There are 428 mean residuals and 423 auxiliary regression observations. Here p+q=0; the reference distribution is chi-square with 5 degrees of freedom.',
             'Residual Ljung-Box uses lag 10 and reference degrees of freedom 10-p-q. The old code omitted this correction. [4,5]',
             'ARCH LM rejection supports modeling conditional variance. Reference p-values after fitting and model selection are approximate.'],y)
    new('Training diagnostics: actual computed results')
    table(['Diagnostic','Statistic','p-value'],[
        ['Residual Ljung-Box, lag 10',f"{d['arma_residual_ljung_box']['Q']:.4f}",f"{d['arma_residual_ljung_box']['p_value']:.5g}"],
        ['Squared residual Ljung-Box, lag 10',f"{d['arma_residual_squared_ljung_box']['Q']:.4f}",f"{d['arma_residual_squared_ljung_box']['p_value']:.5g}"],
        ['ARCH LM, 5 lags',f"{d['arch_lm_5']['statistic']:.4f}",f"{d['arch_lm_5']['p_value']:.5g}"],
        ['Jarque-Bera, mean residuals',f"{d['jarque_bera_mean_residuals']['statistic']:.4f}",f"{d['jarque_bera_mean_residuals']['p_value']:.5g}"]],[510,160,194])
    para('Raw-residual Ljung-Box is not heteroskedasticity-robust. Its large p-value is not proof of independent returns. Jarque-Bera concerns the residual distribution, not conditional Gaussianity after volatility modeling.',145,'small')
    new('ARCH and GARCH: what is being modeled?')
    y=equation(r'\epsilon_t=\sqrt{h_t}z_t,\qquad E(z_t\mid\mathcal{F}_{t-1})=0,\quad E(z_t^2\mid\mathcal{F}_{t-1})=1')
    y=equation(r'h_t=\omega+\alpha\epsilon_{t-1}^2+\beta h_{t-1}',y)
    bullets(['Constant variance: alpha=beta=0. ARCH(1): beta=0. GARCH(1,1): both terms may vary.',
             'Impose omega>0, alpha,beta>=0, alpha+beta<=0.999. Under the usual innovation assumptions this ensures finite unconditional variance; it is a sufficient condition, not the general strict-stationarity criterion.',
             'The Gaussian expression is a quasi-likelihood criterion; it does not assert normally distributed returns.'],y)
    new('Corrected variance estimation and comparison')
    y=equation(r'\ell_v=-\frac{1}{2}\sum_{t=2}^{428}\left[\log(2\pi)+\log(h_t)+\frac{\widehat\epsilon_t^2}{h_t}\right]')
    bullets(['Two-stage estimation: hold the ARMA mean fit fixed, then fit each variance model to the same residual sequence.',
             'Use one common training backcast h(1)=average residual square. Omit its density for every model, giving 427 scored observations.',
             'Retain all 428 mean residuals. The old arbitrary ten-residual trim was justified incorrectly as diffuse initialization.',
             'Compare conditional variance BIC with k=1,2,3. Accept only successful, finite, feasible optimizations; record every optimizer attempt.'],y)
    new('Variance coefficients and training selection')
    y=table(['Model','omega','alpha','beta','Conditional BIC'],[
        [k,f"{a['omega']:.6f}",f"{a['alpha']:.6f}",f"{a['beta']:.6f}",f"{a['bic']:.4f}"] for k,a in v.items()],[190,150,150,150,224])
    g=v['GARCH(1,1)']
    bullets([f"Training BIC selects {r['selected_variance_by_training_bic']}; this choice precedes comparison of test losses.",
             f"GARCH alpha+beta={g['alpha']+g['beta']:.6f}; implied long-run variance is omega/(1-alpha-beta)={g['omega']/(1-g['alpha']-g['beta']):.6f} squared percentage points.",
             'These BIC values compare variance models conditional on the same fitted mean. They must not be compared directly with the ARMA table.'],y)
    new('Forecast variance before observing the return')
    y=equation(r'\widehat h_{t|t-1}=\widehat\omega+\widehat\alpha e_{t-1}^2+\widehat\beta\widehat h_{t-1|t-2}')
    bullets(['The first test variance uses the last training residual and training variance state.',
             'For each test day: save its mean and variance forecasts first; only then use its observed return to form the error.',
             'That error updates the next forecast. It never enters the forecast for its own day. [6]',
             'This is 108 successive one-step forecasts, not a single 108-step forecast made at the training cutoff.'],y)
    new('Volatility scoring: the important qualification')
    y=equation(r'\mathrm{QLIKE}=\frac{1}{108}\sum_t\left[\log(\widehat h_t)+e_t^2/\widehat h_t\right]')
    y=equation(r'E(e_t^2\mid\mathcal{F}_{t-1})=\mathrm{Var}(Y_t\mid\mathcal{F}_{t-1})+(m_t-\widehat Y_{t|t-1})^2',y)
    para('Here m(t)=E[Y(t)|F(t-1)]. Thus squared forecast error is an unbiased conditional-variance proxy only when the forecast mean is correct. All variance candidates share the same mean forecast.',y,'small')
    para('Lower average QLIKE is better on this sample. The secondary variance-proxy MSE is average (e(t)^2-h(t))^2, in fourth powers of percentage points. QLIKE values depend on the return scale; comparisons here use one common scale.',112,'small')
    new('Holdout results: mean forecasting')
    y=table(['Model','RMSE (pp)','MAE (pp)'],[[k,f"{a['rmse']:.6f}",f"{a['mae']:.6f}"] for k,a in mm.items()],[440,212,212])
    bullets(['Zero return has the lowest observed RMSE. The historical mean / ARMA(0,0) has slightly lower MAE.',
             'ARMA(0,0) and the historical mean are the same constant-mean model in theory; tiny numerical differences arise from optimization.',
             'No statistically significant forecast superiority is claimed: this report does not perform a forecast-comparison significance test.'],y)
    new('Holdout results: variance forecasting')
    y=table(['Model','QLIKE','Variance-proxy MSE'],[[k,f"{a['qlike']:.6f}",f"{a['variance_proxy_mse']:.6f}"] for k,a in vm.items()],[360,232,272])
    bullets(['GARCH(1,1), already selected by training BIC, has the lowest observed test QLIKE and variance-proxy MSE among these three candidates.',
             'The result supports this fitted volatility model on this historical split. It does not establish universal superiority or trading profitability.',
             'Updated scores differ from the old PDF because the unnecessary residual trim and inconsistent first-observation likelihood treatment were corrected.'],y)
    new('Standardized residuals: check the variance fit')
    y=equation(r'\widehat z_t=\widehat\epsilon_t/\sqrt{\widehat h_t}')
    y=table(['Diagnostic (first backcast excluded)','Q, lag 10','Reference p'],[
        ['Standardized residuals',f"{d['garch_stdresid_ljung_box']['Q']:.5f}",f"{d['garch_stdresid_ljung_box']['p_value']:.5f}"],
        ['Squared standardized residuals',f"{d['garch_stdresid_squared_ljung_box']['Q']:.5f}",f"{d['garch_stdresid_squared_ljung_box']['p_value']:.5f}"]],[500,180,184],y)
    para('The ACFs show reduced residual-square dependence. These ordinary reference p-values are descriptive: they are not a specially adjusted GARCH portmanteau test. A large p-value does not prove adequacy, independence or Gaussianity.',y,'small')
    charts=[
        ('Observed USD-based ISE returns','01_returns.png','The full-series plot is descriptive. Model choices use only the training block before the dashed cutoff.'),
        ('Training return distribution','02_distribution.png','The Gaussian overlay is a visual comparison; it does not impose a Gaussian data-generating distribution.'),
        ('Training return ACF','03_acf.png','ACF bands are conventional approximate pointwise bands. They are not simultaneous or heteroskedasticity-robust.'),
        ('Squared centered-return ACF','04_squared_acf.png','Dependence in squared returns can coexist with little linear dependence in the returns themselves.'),
        ('Training return PACF','05_pacf.png','PACF guides inspection; the final mean order is chosen by training BIC.'),
        ('Rolling standard deviation','06_rolling_sd.png','The 20-observation rolling sample SD is descriptive and is not used as an external model predictor.'),
        ('ARMA order comparison','07_aic_bic.png','All converged candidates use the same training returns. AIC and BIC are displayed; BIC determines the order.'),
        ('Test one-step mean forecasts','08_forecast_returns.png','A nearly flat forecast is expected because training BIC selected ARMA(0,0); it is not a plotting failure.'),
        ('Test mean forecast errors','09_forecast_errors.png','Each error is computed after its corresponding forecast has been saved.'),
        ('Mean-model residual ACF','10_residual_acf.png','Failure to detect linear autocorrelation does not establish independent errors.'),
        ('Mean-model squared-residual ACF','11_residual_squared_acf.png','Remaining squared-residual dependence motivates a conditional-variance model.'),
        ('Mean-model residual normal Q-Q','12_residual_qq.png','This checks unstandardized mean residuals. Non-Gaussian shape can partly reflect changing conditional variance.'),
        ('Test conditional standard deviation','13_volatility_forecast.png','Absolute forecast error is shown for context; it is not the observed true volatility.'),
        ('GARCH standardized residuals','14_standardized_residual.png','Residuals are divided by fitted training conditional SD. The first value uses the fixed training backcast.'),
        ('Mean forecasting: RMSE comparison','15_mean_comparison.png','Zero return wins RMSE on this test block. MAE has a different ranking; both are reported.'),
        ('Variance forecasting: QLIKE comparison','16_variance_comparison.png','All candidates share the same 108 forecast errors and return scale. This is an observed loss comparison.'),
        ('Standardized-residual ACF','17_garch_std_acf.png','The first backcast observation is excluded. Conventional ACF bands remain descriptive after model fitting.'),
        ('Squared standardized-residual ACF','18_garch_std_squared_acf.png','Inspect whether volatility dependence remains; the ordinary bands are not a formal adjusted GARCH diagnostic.'),
        ('Standardized-residual normal Q-Q','19_garch_std_qq.png','This distribution check is appropriate after variance standardization; normality is not required merely to compute Gaussian QML.')]
    for title,name,note in charts:plot_page(title,name,note)
    new('Audit corrections: code and statistical claims')
    table(['Original issue','Correction'],[
        ['Guessed column; dropped malformed/duplicate rows','Strict header and date checks; reject damaged or incomplete data'],
        ['Finite optimizer output could be selected after failure','Require successful convergence and feasible parameters'],
        ['Ten residuals removed; unequal initial likelihood treatment','Retain 428 residuals; common backcast; score the same 427 densities'],
        ['KPSS 0.1 and ADF 0 displayed as exact p-values','KPSS >0.1; ADF numerical zero reported as <1e-16'],
        ['Incomplete diagnostics and missing slide-builder dependency','19 plots, summary tables, model summary, and PDF generated by this single script']],[410,454])
    new('Reproduce this exact workflow')
    bullets(['Install the supplied requirements.txt in a Python environment. Package versions used for this run are recorded in results.json.',
             'Run: python run_ise_project.py --data data_akbilgic.xlsx --out results --pdf',
             'The original ZIP, the ten-column ISE_full_original.csv and the two-column used_original_univariate_series.csv are also accepted as inputs.',
             'Run: python run_ise_project.py --self-test. Numerical checks cover likelihood scoring and forecast alignment. The delivered archive includes an additional audit validation log.'])
    new('Source record and reproducibility boundary')
    y=para('Source: Akbilgic, O. (2013), <i>ISTANBUL STOCK EXCHANGE</i>, UCI Machine Learning Repository. DOI: 10.24432/C54P4J. License: CC BY 4.0.',405)
    y=para('SHA-256 of the supplied workbook: '+s['source_sha256'],y,'small')
    bullets(['The supplied ZIP contains the original-named workbook with the expected two-row header. Values exported to CSV were checked cell by cell against this workbook.',
             'The former GitHub mirror is no longer needed. Matching old rounded scores is not a byte-for-byte audit of that unavailable mirror.',
             'ADF/KPSS, ARMA order and all parameter fits use training observations. Full-data and test plots are retrospective descriptions.'],y)
    new('References')
    refs=[
        ('[1] UCI dataset and original workbook','https://archive.ics.uci.edu/dataset/247/istanbul+stock+exchange'),
        ('[2] statsmodels: KPSS p-value bounds','https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html'),
        ('[3] statsmodels: ARIMA parameterization','https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html'),
        ('[4] statsmodels: Ljung-Box degrees of freedom','https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html'),
        ('[5] statsmodels: ARCH LM degrees-of-freedom correction','https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_arch.html'),
        ('[6] arch: one-step variance forecasting','https://bashtage.github.io/arch/doc/univariate/forecasting.html')]
    y=410
    for label,url in refs:
        y=para('<b>'+label+'</b><br/><link href="'+url+'" color="#176B85">'+url+'</link>',y,'small')
    new('Final conclusion: mean and volatility differ')
    bullets(['<b>Mean:</b> training BIC selects ARMA(0,0). This gives no selected lagged mean dynamics. Zero return has lower test RMSE; the constant mean has slightly lower MAE.',
             '<b>Volatility:</b> ARCH effects motivate a variance model. Training BIC selects GARCH(1,1), which also achieves the lowest test QLIKE of the three candidates.',
             '<b>Interpretation:</b> the evidence for modeling volatility is stronger here than the evidence for forecasting the conditional mean from past returns.',
             '<b>Limit:</b> one short, previously examined historical split does not establish statistical superiority, future performance or profitable trading.'])
    c.save()
    print(f'Generated {page}-page PDF: {destination}')
    return destination


def self_test():
    """Deterministic forecast alignment and QML-likelihood checks; no fake metrics."""
    e=np.array([.2,-1.,.8,.1,-.4,1.2])
    f=qml_fit(e,'constant')
    assert np.isclose(f['omega'],np.mean(e[1:]**2))
    assert f['n_effective']==len(e)-1
    for name in ['ARCH(1)','GARCH(1,1)']:
        f=qml_fit(e,name)
        a=variance_forecasts(f,e,np.array([2.,4.,6.]))
        b=variance_forecasts(f,e,np.array([9.,4.,6.]))
        assert a[0]==b[0] # today's realization cannot affect today's forecast
        expected=f['omega']+f['alpha']*e[-1]**2+f['beta']*f['variance'][-1]
        assert np.isclose(a[0],expected)
        h=np.array(f['variance'])[1:]
        assert np.isclose(f['loglik'],-.5*np.sum(np.log(2*np.pi*h)+e[1:]**2/h))
    print('SELF TEST OK: QML likelihood, common scoring sample, and forecast alignment')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=str,help='Path to original UCI data_akbilgic.xlsx or a faithful CSV copy')
    parser.add_argument('--download',action='store_true',help='Download original UCI dataset archive')
    parser.add_argument('--out',default='ise_results',help='Output directory')
    parser.add_argument('--self-test',action='store_true',help='Test pipeline only (synthetic data)')
    parser.add_argument('--pdf',action='store_true',help='Build the corrected slide-style PDF from this run')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:
        if not args.data and not args.download:parser.error('Supply --data or --download')
        result=run('__download__' if args.download else args.data,args.out)
        if args.pdf: build_pdf(result,Path(args.out))
