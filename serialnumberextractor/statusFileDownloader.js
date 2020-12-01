var Q = require("q");
var request = require("request");
var mkdirp = require('mkdirp');
var fs = require("fs");
var _ = require('lodash');
var util = require("./util");
var async = require("async");
var fs = require('fs');
var tsdrApiUrl = "https://tsdrsec.uspto.gov/ts/cd/caseMultiStatus/sn?ids=";
var chunkSize = 25;
var logger = util.getLogger();
var downloadDir = '/Users/aaratrikanula/Desktop/APICODE-changes/json/';
var subfolder;

function downloadFiles(serialNumbers, folder){
    subfolder = folder;
    logger.info("Starting task [downloadFiles]");
    var deferred = Q.defer();
    // we will download files in batches of 25
    var serialNumberChunks = createSerialNumberChunks(serialNumbers, chunkSize);
    var unprocessedSerialNumbers = [];
    var errors = [];
    // can only make one request at a time, that too with a delay between each request. Otherwise, it results in error
    var q = async.queue(processChunkWithDelay, 1);
    q.drain = function(){
        var failureCount = unprocessedSerialNumbers.length;
        var failurePercentage = Math.round((failureCount/serialNumbers.length)*100);
        logger.info("Ending task [downloadFiles]");
        if(failurePercentage > 10 || errors.length > 4){
            deferred.reject(errors);
        }else{
            deferred.resolve(unprocessedSerialNumbers);
        }
    };
    serialNumberChunks.forEach(function(serialNumbers, index){
        q.push({
            serialNumbers: serialNumbers,
            chunkNumber:index
        }, function(err, errorList){
            if(err){
                errors.push(err);
            }else{
                unprocessedSerialNumbers.push.apply(unprocessedSerialNumbers, errorList);
            }
        });
    });
    return deferred.promise;
}

function processChunkWithDelay(obj, callback){
    // got to give enough timeout between each request, otherwise we are seeing connection errors
    setTimeout(function(){ processChunk(obj, callback)}, 1000);
}

// returns a list of serial numbers that couldn't be downloaded
function processChunk(obj, callback){
    var serialNumbers = obj.serialNumbers;
    var chunkNumber = obj.chunkNumber;
    logger.info("Processing chunk number - "+chunkNumber);
    var url = tsdrApiUrl+serialNumbers.join(",");
    var processedCount = 0;
    request(url, function(err, response, data){
        if(!err && response.statusCode == 200) {
            var transactionList = JSON.parse(data).transactionList;
            // errorList has list of serial numbers that were not retrieved from the API or that couldn't be written to the file system
            var errorList = getMissingSerialNumbers(serialNumbers, transactionList);
            var totalTransactions = transactionList.length;
            transactionList.forEach(function(transaction){
                var serialNumber = transaction.searchId;
               // var path = serialNumber.slice(0, 2)+"/"+serialNumber.slice(2, 4)+"/"+serialNumber.slice(4,6);
                var fullPath = downloadDir;
                mkdirp(fullPath, function(err){
                    if(!err){
                        var filePath = fullPath+"/"+serialNumber+".json";
                        fs.writeFile(filePath, JSON.stringify(transaction), function(err, response){
                            processedCount++;
                            if(processedCount == totalTransactions){
                                callback(null, errorList);
                            }
                        })
                    }else{
                        errorList.push(serialNumber);
                        logger.info("Error processing chunks: ",err);
                        processedCount++;
                        if(processedCount == totalTransactions){
                            callback(null, errorList);
                        }
                    }
                })
            })
        }else{
            logger.info("Error processing chunks: ",err);
            logger.error(err);
            callback(err);
        }
    });
}

function getMissingSerialNumbers(serialNumbers, transactionList){
    var serialNumbersFromAPI =  _.map(transactionList, "searchId");
    var missingSerialNumbers = _.difference(serialNumbers, serialNumbersFromAPI)
    return missingSerialNumbers;
}


function createSerialNumberChunks(array, chunkSize) {
    var results = [];
    var i , j ;
    for (i=0,j=array.length; i<j; i+=chunkSize) {
        results.push(array.slice(i,i+chunkSize));
    }
    return results;
};

exports.downloadFiles = function(serialNumbers, folder){
    return downloadFiles(serialNumbers, folder);
}


setTimeout(function () {
    var textByLine = fs.readFileSync('/Users/NewTextSNs_2020-08-31_2020-10-06.txt').toString();
      var serialNumbers = textByLine.split("\n"); //textByLine.split(",");
      console.log(serialNumbers);
      downloadFiles(serialNumbers, "").then(function(errorList){
          console.log(errorList);
          console.log("Total Error Count- "+errorList.length);
      }, function(err){
          console.log("Error");
          console.log(err);
      });
  }, 1000)


