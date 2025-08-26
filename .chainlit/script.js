// 专门处理用户邮箱显示的脚本
(function() {
    'use strict';
    
    function replaceEmailWithUsername() {
        // 查找所有文本节点
        const walker = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );
        
        let textNode;
        const textNodes = [];
        
        // 收集所有文本节点
        while (textNode = walker.nextNode()) {
            textNodes.push(textNode);
        }
        
        // 替换邮箱文本
        textNodes.forEach(node => {
            if (node.textContent.includes('me@example.com')) {
                node.textContent = node.textContent.replace('me@example.com', '当前用户');
            }
        });
    }
    
    // 页面加载完成后执行
    function init() {
        replaceEmailWithUsername();
        
        // 监听DOM变化
        const observer = new MutationObserver(function(mutations) {
            let shouldUpdate = false;
            mutations.forEach(mutation => {
                if (mutation.type === 'childList' || mutation.type === 'characterData') {
                    shouldUpdate = true;
                }
            });
            
            if (shouldUpdate) {
                setTimeout(replaceEmailWithUsername, 100);
            }
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            characterData: true
        });
    }
    
    // 确保在DOM准备好后执行
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();