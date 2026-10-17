// Frontend fixtures only. Never returned by the analysis API.
export const demoScenarios = {
  events: {
    verdict: 'violation',
    violations: [
      {
        type: 'Предмет у края рабочей зоны',
        timecode_sec: 12.4,
        description: 'На учебной схеме предмет A расположен у границы рабочей поверхности. Переключите изображения, чтобы сравнить его положение. Это вымышленное событие для проверки интерфейса, а не вывод о нарушении.',
        evidence_frames: [
          { timestamp_sec: 12.4, demo_image_url: '/assets/demo/event-1-a.svg' },
          { timestamp_sec: 13.2, demo_image_url: '/assets/demo/event-1-b.svg' },
        ],
      },
      {
        type: 'Предмет перемещён в другую зону',
        timecode_sec: 44.2,
        description: 'Две учебные схемы показывают предмет B в разных частях рабочей поверхности. В реальном результате здесь появится описание наблюдаемого действия от модели. Реальная запись не анализировалась.',
        evidence_frames: [
          { timestamp_sec: 44.2, demo_image_url: '/assets/demo/event-2-a.svg' },
          { timestamp_sec: 45.0, demo_image_url: '/assets/demo/event-2-b.svg' },
        ],
      },
    ],
  },
  empty: { verdict: 'no_violation', violations: [] },
};
