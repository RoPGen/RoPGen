import jpype
from jpype import *
import os

# 获取jvm路径
jvmPath = jpype.getDefaultJVMPath()
print(jvmPath)
#jvmPath = "/usr/lib/jvm/java-11-openjdk-amd64/lib/server/libjvm.so"
# jvmPath = "D:/Java/jre1.8.0_151/bin/server/jvm.dll"
'''
#动态加载jar包
libPath = "/home/zss/data/project5/authorship-detection-master/attribution/pathminer"
lib_jar=['extract-path-contexts.jar']
lib_jar = [ os.path.join(libPath,x) for x in lib_jar]

jvmArg = "-Djava.class.path="+";".join(lib_jar)
print(jvmArg)

#启动jvm
if not jpype.isJVMStarted():
    jpype.startJVM(jvmPath, jvmArg)

#python 执行java代码
jpype.java.lang.System.out.println("helloworld!")

#获取java类。这个是以jar包中的相对路径来找到
JDClass = jpype.JClass("jt")
#这里的有网上说是jd = JDClass()。问题来了，加上括号，在执行函数时可能会失败
jd = JDClass
jd.run()
print( jd.formatDuring(555555) ) 
# main函数的参数是一个list
jd.main(['a'])
'''

 
jarpath = os.path.join(os.path.abspath('.'), '/home/zss/data/project5/authorship-detection-master/attribution/pathminer/')  
startJVM("/usr/lib/jvm/java-11-openjdk-amd64/lib/server/libjvm.so","-ea", "-Djava.class.path=%s" % (jarpath + 'extract-path-contexts.jar'))  
#ubuntu 中startJVM("/home/geek/Android/jdk1.6.0_43/jre/lib/i386/server/libjvm.so","-ea", "-Djava.class.path=%s" % (jarpath + 'XXX.jar'))  
JDClass = JClass("jpype.Extract-path-contexts")  
jd = JDClass()  
#jd = JPackage("jpype").JpypeDemo() #两种创建jd的方法  
jprint = java.lang.System.out.println("hellojava")
#jprint(jd(snapshot project="datasets/java40/" output="processed/java40/" java-parser="antlr" maxContexts=1000 maxL=8 maxW=3))  
shutdownJVM() 