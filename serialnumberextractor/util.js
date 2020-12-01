var winston = require("winston");
var fs = require("fs");

var logs_dir = process.env.LOGS_DIR || "logs";
if (!fs.existsSync(logs_dir)){
    fs.mkdirSync(logs_dir);
}
// either create the files, or empty them
fs.closeSync(fs.openSync(logs_dir+"/debug.log", 'w'));
fs.closeSync(fs.openSync(logs_dir+"/error.log", 'w'));

var logger = new winston.Logger({
    level: 'debug',
    transports: [
        new (winston.transports.File)({name: "debug_file", filename: logs_dir+'/debug.log', level: 'debug', json: false }),
        new (winston.transports.File)({name: "error_file", filename: logs_dir+'/error.log', level: 'error', json: false }),
        new (winston.transports.Console)()
    ],
    exceptionHandlers: [
        new (winston.transports.File)({name: "error_file", filename: logs_dir+'/error.log', level: 'error', json: false }),
        new (winston.transports.Console)()
    ]
});

exports.getLogger = function(){
    return logger;
}

exports.getRandomString = function(length){
    var text = "";
    var possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    for( var i=0; i < length; i++ ){
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}

exports.writeArrayToFile = function(array, filePath, append, callback){
    console.log("Writing to file ["+filePath+"]- total items - "+array.length)
    var stream;
    if(append){
        stream = fs.createWriteStream(filePath, {flags: 'a'});
    }else{
        stream = fs.createWriteStream(filePath);
    }
    stream.once('open', function(fd) {
        array.forEach(function(arrayItem){
            stream.write(arrayItem+"\n");
        })
        stream.end();
        if(callback){
            callback();
        }
    });
}

exports.readLines = function(filePath, callback){
    fs.readFile(filePath, "utf8", function(err, data){
        var lines = data.match(/[^\r\n]+/g);
        callback(null, lines);
    })
}