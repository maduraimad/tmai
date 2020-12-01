var cookieParser = require('cookie-parser');
var bodyParser = require('body-parser');
var request = require("request");
request = request.defaults({jar: true})
var cors = require('cors');
var Q = require("q");
var cheerio = require("cheerio");
var async = require("async");
var util = require("./util");

var logger = util.getLogger();
var tmBaseUrl = "http://tmsearch.uspto.gov/bin/gate.exe?f=login&p_lang=english&p_d=trmk";
var tmStateUrl = "http://tmsearch.uspto.gov/bin/gate.exe?f=tess&state=";
var tmSearchUrl = "http://tmsearch.uspto.gov/bin/gate.exe";
var tmJumpToUrl = "http://tmsearch.uspto.gov/bin/jumpto";
var pageSize = 1000;

//var cron = require('node-cron')

function formatDateForSearch(date){
    var dateString = date.toISOString().slice(0, 10);
    return dateString.split("-").join("");
}

function searchSerialNumbers(searchString){
    logger.info("Starting task [SearchSerialNumbers]");
    logger.info("Using searchString - "+searchString);
    var deferred = Q.defer();
    var deferred = Q.defer();
    var requestOptions = {
        url: tmBaseUrl,
        followRedirect: false
    }
    request(requestOptions, function(err, response, data){

        if(!err && response.statusCode == 302) {
            // the initial request would return a state that needs to be used for establishing a session
            var location = response.headers['location'];
            var state = location.substring(location.lastIndexOf("=")+1);
            var cookies = response.headers["set-cookie"];
            var jar = request.jar();
            var newCookies = [];
            if(cookies){
                cookies.forEach(function(cookie){
                    jar.setCookie(request.cookie(cookie));
                })
            }
            var url = tmStateUrl+state;
            request({url: url}, function(err, response, data){
                if(!err && response.statusCode == 200) {
                    // now we have established a session
                    // var searchString = '(`FD >= '+startDateFormatted+' <= '+endDateFormatted+') OR (`UD >= '+startDateFormatted+' <= '+endDateFormatted+' AND `FD >= 20140101)';
                    logger.info("Using Tess Search String - "+searchString);
                    performSearch(searchString, state).then(function(serialNumbers){
                        logger.info("Ending task [SearchSerialNumbers]");
                        deferred.resolve(serialNumbers);
                    }, function(err){
                        logger.info("Ending task [SearchSerialNumbers]");
                        deferred.reject(err);
                    });
                }
            })
        }else{
            logger.info("Ending task [SearchSerialNumbers]");
            deferred.reject(err);
        }

    });

    return deferred.promise;
}

function performSearch(searchString, state){
    var deferred = Q.defer();
    var newCookies = [];
    var jar = request.jar();
    var queryCookie = encodeURIComponent(searchString);
    jar.setCookie("queryCookie="+queryCookie+"; path=/");
    newCookies.push("queryCookie="+queryCookie);
    var cookieString = newCookies.join(";");
    var searchParams = {
        url: tmSearchUrl,
        followRedirect: false,
        form:{
            f: "toc",
            p_search: "search",
            p_L: pageSize,
            p_plural: "yes",
            p_s_ALL: searchString,
            a_search: "Submit Query",
            state: state
        },
        headers:{
            Cookie: cookieString
        }
    }
    var serialNumbers = [];
    request.post(searchParams, function(err, response, data){
        if(!err && response.statusCode == 200) {
            logger.info("Got the first chunk");
            const $ = cheerio.load(data);
            var serialNumbers = [];
            // parse the returned HTML to retrieve serial numbers
            // if there are no results, just end it here
            if(!($('table').length >= 2)){
                deferred.resolve(serialNumbers);
            }else{
                $($('table').get(-2)).find('tr').each(function (index, item) {
                    serialNumbers.push($($(item).find('td')[1]).children().html())
                });
                // *** remove this line if you want to fetch more pages. Also uncomment two fetchNextBatch() calls
                // deferred.resolve(serialNumbers);
                // fetch rest of the pages in batch
                // each new page uses the state returned in previous batch
                var newState = $("table form input[name='state']").val();
                var fetchNextBatch = function(searchState, jumpTo){
                    logger.info("Serial numbers so far " +serialNumbers.length);
                    logger.info("Processing Starting Value - "+jumpTo);
                    var queryString = {
                        f: "toc",
                        state: searchState,
                        jumpto: jumpTo
                    }
                    var requestOptions = {
                        url : tmJumpToUrl,
                        qs: queryString,
                        headers:{
                            Cookie: cookieString,
                            Referer:'http://tmsearch.uspto.gov/bin/gate.exe',
                            Accept:'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                            'Cache-Control':'no-cache',
                            "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36"
                        }
                    }
                    request(requestOptions, function(err, response, data){
                        if(!err && response.statusCode == 200) {
                            const $ = cheerio.load(data);
                            // parse the returned HTML to retrieve serial numbers
                            // if there are no results
                            if(!($('table').length >= 2)){
                                deferred.resolve(serialNumbers);
                            }else{
                                $($('table').get(-2)).find('tr').each(function(index,item){
                                    serialNumbers.push($($(item).find('td')[1]).children().html());
                                });
                                var newState = $("table form input[name='state']").val();
                                fetchNextBatch(newState, jumpTo+pageSize);
                            }
                        }else{
                            logger.error(err);
                            deferred.reject(err);
                        }
                    })
                };
                fetchNextBatch(newState, pageSize + 1);
            }
        }else{
            deferred.reject(err);
        }
    });
    return deferred.promise;

}

exports.searchSerialNumbers = function(startDate, endDate){
    return searchSerialNumbers(startDate, endDate);
}

exports.searchByCriteria = function(searchCriteria){
    var searchString = ""
    if (searchCriteria instanceof  Array){
        searchTokens = []
        searchCriteria.forEach(function(criteria){
            searchTokens.push("("+buildSearchString(criteria)+")");
        })
        searchString = searchTokens.join(" AND ")
    }else{
        searchString = buildSearchString(searchCriteria);
    }
    console.log(searchString)
    return searchSerialNumbers(searchString);
}

function buildSearchString(searchCriteria){
    var searchType = searchCriteria.searchType;
    var searchString;
    if(searchType == "fd" || searchType == "fdud"){
        var startDateFormatted = formatDateForSearch(searchCriteria.startDate);
        var endDateFormatted = formatDateForSearch(searchCriteria.endDate);
        if(searchType == "fdud"){
            searchString = '(`FD >= '+startDateFormatted+' <= '+endDateFormatted+') OR (`UD >= '+startDateFormatted+' <= '+endDateFormatted+' AND `FD >= 20140101)';
        }else{
            searchString =  '(`FD >= '+startDateFormatted+' <= '+endDateFormatted+')';
        }
        searchString = searchString  // + " AND `TD > 0" // Enable for getting images with design codes
    }else if(searchType == "fmexact"){
        var markName = searchCriteria.markName;
        var fullMarkSearch = markName.split(" ").join("-");
        searchString = fullMarkSearch+"[FM]";
    }else if(searchType == "fm"){
        var markSearchTokens = [];
        var markName = searchCriteria.markName;
        markSearchTokens.push(markName.split(" ").join("-")+"[FM]");
        markName.split(" ").forEach(function(markNameToken){
            markSearchTokens.push(markNameToken+"[FM]")
        })
        searchString = markSearchTokens.join(" OR ");
    }else if(searchType == "md"){
        var descriptionSearchTokens = [];
        var markDescription = searchCriteria.markDescription;
        markDescription.split(" ").forEach(function(descriptionToken){
            descriptionSearchTokens.push(descriptionToken+"[DE]")
        })
        searchString = descriptionSearchTokens.join(" AND ");
    }else if(searchType == "dc"){
        searchString = searchCriteria.designCode+"[DC]";
    }else if(searchType == "on"){
        searchString = "\""+searchCriteria.ownerName+"\"[ON]"
    }else if(searchType == "mdc"){
        var markDrawingCodes = searchCriteria.markDrawingCodes;
        var tokens = []
        markDrawingCodes.forEach(function(code){
            tokens.push(code+"[MD]")
        })
        searchString = tokens.join(" OR ")
    }else if(searchType == "gs"){
        searchString = "\""+searchCriteria.ownerName+"\"[GS]"
    }

    return searchString;
}

function getSerialNumbersForOwners(){
    logger.info ("********Inside getSerialNumbersForOwners()**********")
    var owners = ["Adidas"];
    var promiseList = [];
    var allSerials = [];
    owners.forEach(function(owner){
        var searchCriteria = {
            searchType: "on",
            ownerName: owner
        }
        var searchString = buildSearchString(searchCriteria);
        logger.info ("********searchString**********",searchString)
        promiseList.push(searchSerialNumbers(searchString));
    })
    Q.allSettled(promiseList).then(function(results){
        results.forEach(function (result, index) {
            if (result.state === "fulfilled") {
                logger.info("Successfully obtained serial numbers for -"+owners[index]);
                logger.info("Total serials for "+owners[index]+"- "+result.value.length);
                allSerials.push.apply(allSerials, result.value);
            } else {
                logger.error("Error in obtainingserial numbers for -"+owners[index]);
            }
        });
        util.writeArrayToFile(allSerials, "/Users/greensod/usptoWork/TrademarkRefiles/data/ownerNameSerials.txt", true);
    })
}


// From 1350 design codes, we are trying to filter the ones that have less than 200 serial numbers
function filterDesignCodesWithLimitedSNs(){
    final_list = []
    tasks = []
    //    /Users/greensod/usptoWork/TrademarkRefiles/data/1382-designCodes.txt
    util.readLines("/Users/greensod/temp/delete/1382-designCodes.txt", function(err, design_codes){
        design_codes.forEach(function(design_code, index){
            tasks.push((function(local_design_code, local_index){
                return function(callback){
                    logger.info("Processing index - "+local_index+" design code - "+local_design_code)
                    var searchCriteria = {
                        searchType: "dc",
                        designCode: local_design_code
                    }
                    exports.searchByCriteria(searchCriteria).then(function(serials){
                        if(serials && serials.length < 200){
                            final_list.push(local_design_code)
                        }
                        callback(null)
                    }, function(err){
                        logger.error("Error in processing desing code - "+local_design_code)
                        callback(null)
                    })
                }
            })(design_code, index))
        })
        logger.info("Total tasks - "+tasks.length)
        async.series(tasks, function(err, result){
            logger.info("All done. Printing results ");
            final_list.forEach(function(item){
                console.log(item.substr(0,2)+"."+item.substr(2,2)+"."+item.substr(4,2))
            })
        })
    })
}


setTimeout(function () {
    var searchCriteria = {
        searchType: "fd",
        startDate: new Date("2020-10-06"),
        endDate: new Date("2020-11-06")
    }
    module.exports.searchByCriteria(searchCriteria).then(function(serialNumbers){
        logger.info("Total Count - "+serialNumbers.length);
        util.writeArrayToFile(serialNumbers, "/Users/NewTextSNs_2020-08-31_2020-10-06.txt", false, function(err){
            if (!err){
                log.info("Done")
            }else{
                log.info("Done with error")
            }
        })
    }, function(err){
        logger.error(err);
    });
}, 500)



//getSerialNumbersForOwners();
//filterDesignCodesWithLimitedSNs()

