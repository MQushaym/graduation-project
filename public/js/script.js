// كود إضافي للتأكد من تحميل ملف التنسيق
var link = document.createElement('link');
link.rel = 'stylesheet';
link.type = 'text/css';
link.href = '/public/css/style.css?v=' + Date.now(); // إضافة رقم عشوائي لكسر الكاش
document.getElementsByTagName('head')[0].appendChild(link);

document.addEventListener('DOMContentLoaded', function() {
    // 1. تغيير أيقونة التبويب (Favicon)
    var link = document.querySelector("link[rel*='icon']") || document.createElement('link');
    link.type = 'image/png';
    link.rel = 'shortcut icon';
    link.href = '/public/assets/Logo.png'; 
    document.getElementsByTagName('head')[0].appendChild(link);

    // 2. تغيير عنوان النافذة من مساعد إلى PharmaGuide
    document.title = "PharmaGuide | المساعد الطبي";
});