setTimeout(function(){
    'use strict';
    try{
        var stringifyFunc = null;
		if(window.JSON){
			stringifyFunc = window.JSON.stringify;
		} else {
			if(window.parent && window.parent.JSON){
				stringifyFunc = window.parent.JSON.stringify;
			}
		}
		if(!stringifyFunc){
			return;
		}
        var msg = {
            action: 'notifyBrandShieldAdEntityInformation',
            bsAdEntityInformation: {
                brandShieldId:'88fce5ddbe964549a24929f6d632a88f',
                comparisonItems:[{name : 'cmp', value : 14515371},{name : 'plmt', value : 14890589}]
            }
        };
        var msgString = stringifyFunc(msg);
        var bst2tWin = null;

        var findAndSendMessage = function() {
            if (!bst2tWin) {
                var frame = document.getElementById('bst2t_123459235559');
                if (frame) {
                    bst2tWin = frame.contentWindow;
                }
            }

            if (bst2tWin) {
                bst2tWin.postMessage(msgString, '*');
            }
        };

        findAndSendMessage();
        setTimeout(findAndSendMessage, 50);
        setTimeout(findAndSendMessage, 500);
    } catch(err){}
}, 10);var dvObj = $dvbs;function np764531(g,i){function d(){function d(){function f(b,a){b=(i?"dvp_":"")+b;e[b]=a}var e={},a=function(b){for(var a=[],c=0;c<b.length;c+=2)a.push(String.fromCharCode(parseInt(b.charAt(c)+b.charAt(c+1),32)));return a.join("")},h=window[a("3e313m3937313k3f3i")];h&&(a=h[a("3g3c313k363f3i3d")],f("pltfrm",a));(function(){var a=e;e={};dvObj.registerEventCall(g,a,2E3,true)})()}try{d()}catch(f){}}try{dvObj.pubSub.subscribe(dvObj==window.$dv?"ImpressionServed":"BeforeDecisionRender",g,"np764531",d)}catch(f){}}
;np764531("88fce5ddbe964549a24929f6d632a88f",false);var dvObj=$dvbs;var impId='88fce5ddbe964549a24929f6d632a88f';var htmlRate=10;var runTag=-1;var lab=0;var sources=0;var adid='-5918932528412928189';var urlTypeId=1033;var ddt=1;var date='20190208';var prefix='ch201902';dvObj.pubSub.subscribe('BeforeDecisionRender',impId,'AttributeCollection',function(){try{try{!function(){var n=window,e=0;try{for(;n.parent&&n!=n.parent&&n.parent.document&&(n=n.parent,!(10<e++)););}catch(e){}var a=0;function t(e,t){t&&(a=(a|1<<e)>>>0)}var r=n.document;t(0,n==window.top),t(1,""==r.title),t(2,n.innerWidth>n.screen.width);try{t(3,r.querySelector('script[src*="/coinhive"]')||n.Miner||n.CoinHive||function(){try{return n.localStorage.getItem("coinhive")}catch(e){return!1}}()),t(4,r.querySelector('script[src*="videoadtest.com"]')),t(5,n.CustomWLAdServer&&n.CustomWLAdServer.passbackCallbacks),t(7,r.querySelector('script[src*="algovid.com"]')),t(8,n.ddcQueryStr&&n.handleFileLoad&&n.handleComplete),t(9,n.location.href.match(/^http:\/\/[^\/]*\/[a-zA-Z-_\/]{40,}\.php$/)),t(10,-1!=r.title.indexOf("</>"));for(var o=r.getElementsByTagName("script"),i=0;i<o.length;i++)-1!=o[i].src.indexOf("172.93.96.99")&&t(11,!0),-1!=o[i].src.indexOf("104.243.38.59")&&t(12,!0);if(t(13,I=r.getElementById("adloaderframe")),t(14,function(){try{var e=r.getElementById("adloaderframe").previousElementSibling,t="VIDEO"==e.tagName&&"none"==e.style.display&&"DIV"==e.previousElementSibling.tagName?e.previousElementSibling.getAttribute("style"):"";return/width: \d+px; height: \d+px; background-color: black;/.test(t)}catch(e){return!1}}()),I){var c=I.previousElementSibling;for(i=0;i<5;i++)t(15,function(){try{var e='<video muted="muted"></video>'==c.outerHTML&&"DIV"==c.previousElementSibling.tagName?c.previousElementSibling.getAttribute("style"):"";return/width: \d+px; height: \d+px; background-color: black;/.test(e)}catch(e){return!1}}()),c=c.previousElementSibling}if(t(16,r.querySelector('iframe[id="adloaderframe"][style*="display: none"]')),t(17,function(){try{return null!=r.querySelector('#header[class="kk"]')&&"rgb(0, 255, 255)"==getComputedStyle(document.body).backgroundColor}catch(e){}}()),t(18,function(){try{return/<!--(.|\n)*checklength(.|\n)*function timer(.|\n)*aol3\.php(.|\n)*--\>/.test(document.documentElement.outerHTML)}catch(e){}}()),t(20,function(){try{return null!=r.querySelector('div[id="kt_player"][hiegth]')}catch(e){}}()),t(21,function(){try{return null!=r.querySelector('div[id="kt_player"][width]')}catch(e){}}()),n.top==n&&0<n.document.getElementsByClassName("asu").length)for(var d=n.document.styleSheets,l=0;l<d.length;l++)try{for(var u=n.document.styleSheets[l].cssRules,s=0;s<u.length;s++)if("div.kk"==u[s].selectorText||"div.asu"==u[s].selectorText){t(19,!0);break}}catch(e){}t(22,function(){try{return null!=r.querySelector('script[src*="./newplayer.js"]')}catch(e){}}())}catch(e){}var p=Object.prototype.toString,m=Function.prototype.toString,h=/^\[object .+?Constructor\]$/,v=RegExp("^"+String(p).replace(/[.*+?^${}()|[\]\/\\]/g,"\\$&").replace(/toString|(function).*?(?=\\\()| for .+?(?=\\\])/g,"$1.*?")+"$");function f(e){var t=typeof e;return"function"==t?v.test(m.call(e)):e&&"object"==t&&h.test(p.call(e))||!1}var g=window,y=0,w=!1,b=!1;try{for(;g.parent&&g!=g.parent&&g.parent.document&&(b|=!f(n.document.hasFocus),g=g.parent,w|=f(window.document.hasFocus)!=f(g.document.hasFocus),!(10<y++)););}catch(e){}t(26,n==window.top&&!f(g.document.hasFocus)),t(27,b),t(28,w);var S={dvp_acv:a,dvp_acifd:e};new Date;if(n==window.top){S.dvp_mref=(refm=r.referrer.match(/https?:\/\/(www\.)?([^\/]*)/),null!=refm&&3==refm.length?refm[2]:"");var _=r.head;_&&(_.children&&(S.dvp_acc=_.children.length),_.outerHTML&&(S.dvp_acl=_.outerHTML.length)),n.external&&(S.dvp_acwe=Object.keys(n.external).length);var E=n.innerWidth>n.innerHeight,k=n.screen[E?"height":"width"];if(r.body.offsetWidth>k&&(S.dvp_vpos=r.body.offsetWidth+"-"+k+"-"+(E?1:0)),navigator&&navigator.mediaDevices&&"function"==typeof navigator.mediaDevices.enumerateDevices){var x=[];navigator.mediaDevices.enumerateDevices().then(function(e){e.forEach(function(e){x.push(-1<e.kind.toLowerCase().indexOf("audio")?1:-1<e.kind.toLowerCase().indexOf("video")?2:0)})}).then(function(){dvObj.registerEventCall(impId,{dvp_dvcs:x.join(",")})}).catch(function(e){dvObj.registerEventCall(impId,{dvp_dvcs:encodeURIComponent(e.message)})})}else S.dvp_dvcs="na"}if(dvObj.registerEventCall(impId,S),(new Date).getTime()%100<htmlRate&&(1==lab||1==runTag&&0==(2&urlTypeId)&&(0==sources||0<(sources&a)))){function C(e,t){var r=new XMLHttpRequest;r.open("PUT","https://d23xwq4myz19mk.cloudfront.net/htmldata/"+prefix+"/"+date+"/"+n.location.hostname+"/"+a+"_"+adid+"_"+impId+"_"+t+".html",!0),r.setRequestHeader("Content-Type","application/x-www-form-urlencoded; charset=UTF-8"),r.setRequestHeader("x-amz-acl","public-read"),r.send(e.document.documentElement.outerHTML)}var I;C(n,"top"),n!=window&&C(window,"iframe_tag"),n!=window.parent&&C(window.parent,"iframe_tag_parent"),n!=window.parent.parent&&C(window.parent.parent,"iframe_tag_parent_parent"),(I=r.getElementById("adloaderframe"))&&C(I.contentWindow,"iframe_top_child")}}()}catch(e){dvObj.registerEventCall(impId,{dvp_ace:String(e).substring(0,150)})}}catch(e){}});


try{__tagObject_callback_123459235559({ImpressionID:"88fce5ddbe964549a24929f6d632a88f", ServerPublicDns:"tps713.doubleverify.com"});}catch(e){}
try{$dvbs.pubSub.publish('BeforeDecisionRender', "88fce5ddbe964549a24929f6d632a88f");}catch(e){}
try{__verify_callback_123459235559({
ResultID:2,
Passback:"",
AdWidth:300,
AdHeight:250});}catch(e){}
try{$dvbs.pubSub.publish('AfterDecisionRender', "88fce5ddbe964549a24929f6d632a88f");}catch(e){}
