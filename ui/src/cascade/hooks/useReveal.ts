import { useEffect } from 'react';

export function useReveal(selector = '.cg-reveal') {
  useEffect(() => {
    const els = Array.from(document.querySelectorAll(selector));
    if (!('IntersectionObserver' in window)) {
      els.forEach(e => e.classList.add('in'));
      return;
    }
    const io = new IntersectionObserver((entries) => {
      entries.forEach(en => {
        if (en.isIntersecting) {
          en.target.classList.add('in');
          io.unobserve(en.target);
        }
      });
    }, { threshold: 0.12 });
    els.forEach(e => io.observe(e));
    return () => io.disconnect();
  }, [selector]);
}
